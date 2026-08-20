"""
阶段3: 查询CatWISE获取真实WISE星等

通过VizieR (II/365/catwise)查询每个PS光学候选体的w1flux/w2flux，
替换preprocessed_cat中CSV的占位值(SDSSxWISE中位数)。

由于SDSSxWISE只覆盖北天(Dec > -19.7)，而训练笔记中的PS源在南天(Dec ~ -28)，
所以需要从CatWISE全天星表重新获取。

输出: 更新后的4个CSV文件
"""

import os
import time
import json
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
PREPROC_DIR = os.path.join(DATA_ROOT, "catalogs", "preprocessed_cat")
LOG_FILE = os.path.join(DATA_ROOT, "logs", "query_catwise_log.txt")
CACHE_FILE = os.path.join(DATA_ROOT, "cache", "catwise_cache.json")

CSV_FILES = [
    "PS_p_Norris06_samples.csv",
    "PS_n_Norris06_samples.csv",
    "PS_p_RGZ_samples.csv",
    "PS_n_samples_RGZ_all.csv",
]

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line.encode('ascii', errors='replace').decode('ascii'))
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def query_catwise_single(ra, dec, radius_arcsec=3.0):
    """Query CatWISE for a single position, return (w1flux, w2flux) or (None, None)"""
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=['mW1', 'mW2', 'nW1', 'nW2'], row_limit=3)
        # CatWISE2020: II/365/catwise
        result = v.query_region(
            SkyCoord(ra, dec, unit='deg', frame='icrs'),
            radius=radius_arcsec * u.arcsec,
            catalog='II/365/catwise'
        )
        if result is None or len(result) == 0:
            return None, None

        table = result[0]
        if len(table) == 0:
            return None, None

        # Pick the one with best W1 detection count (nW1)
        best_idx = 0
        best_n = -1
        for i, row in enumerate(table):
            n = int(row['nW1']) if row['nW1'] is not None else 0
            if n > best_n:
                best_n = n
                best_idx = i

        # Convert Vega magnitudes to flux in mJy
        w1_mag = float(table[best_idx]['mW1']) if table[best_idx]['mW1'] is not None else None
        w2_mag = float(table[best_idx]['mW2']) if table[best_idx]['mW2'] is not None else None

        if w1_mag is not None:
            w1flux = 309.540 * 10**(-w1_mag/2.5) * 1000  # Jy -> mJy
        else:
            w1flux = None

        if w2_mag is not None:
            w2flux = 171.787 * 10**(-w2_mag/2.5) * 1000
        else:
            w2flux = None

        return w1flux, w2flux

    except Exception as e:
        log(f"  Query error: {str(e)[:100]}")
        return None, None

def main():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"CatWISE Query Log\n{'='*60}\n")

    # Load cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        log(f"Loaded {len(cache)} cached entries")

    # ---- Step 1: Collect unique PS sources across all CSVs ----
    log("Step 1: Collecting unique PS sources...")
    all_sources = {}
    for csv_file in CSV_FILES:
        path = os.path.join(PREPROC_DIR, csv_file)
        if not os.path.exists(path):
            log(f"  MISSING: {csv_file}")
            continue
        df = pd.read_csv(path)
        log(f"  {csv_file}: {len(df)} rows")
        for _, row in df.iterrows():
            key = str(row.get('Pan-STARRS_objID', f"{row['Pan-STARRS_RAJ2000']}_{row['Pan-STARRS_DEJ2000']}"))
            all_sources[key] = {
                'ra': float(row['Pan-STARRS_RAJ2000']),
                'dec': float(row['Pan-STARRS_DEJ2000'])
            }

    log(f"\n  Unique PS sources: {len(all_sources)}")
    ra_values = [s['ra'] for s in all_sources.values()]
    dec_values = [s['dec'] for s in all_sources.values()]
    log(f"  RA range: {min(ra_values):.1f} - {max(ra_values):.1f}")
    log(f"  Dec range: {min(dec_values):.1f} - {max(dec_values):.1f}")

    # ---- Step 2: Query CatWISE ----
    log(f"\nStep 2: Querying CatWISE (VizieR II/365)...")
    results = {}
    new_queries = 0
    failed = 0

    for i, (key, info) in enumerate(all_sources.items()):
        if key in cache:
            results[key] = cache[key]
            continue

        ra, dec = info['ra'], info['dec']
        w1flux, w2flux = query_catwise_single(ra, dec)
        new_queries += 1

        if w1flux is not None:
            results[key] = {'w1flux': w1flux, 'w2flux': w2flux if w2flux else w1flux * 0.5}
        else:
            results[key] = {'w1flux': None, 'w2flux': None}
            failed += 1

        # Save cache periodically
        if new_queries % 50 == 0:
            cache.update(results)
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f)
            log(f"  Progress: {new_queries}/{len(all_sources) - len(cache) + new_queries} (failed={failed})")

        # VizieR rate limit: ~1 query/sec
        time.sleep(1.2)

    # Save final cache
    cache.update(results)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

    log(f"\n  Queries: {new_queries}")
    log(f"  With flux: {sum(1 for v in results.values() if v.get('w1flux') is not None)}")
    log(f"  Failed: {failed}")
    log(f"  From cache: {len(cache) - new_queries}")

    # ---- Step 3: Update CSVs ----
    log(f"\nStep 3: Updating CSV files...")

    # Get SDSSxWISE median values as fallback
    median_w1 = 619.5  # from rebuild_catalogs.py
    median_w2 = 713.8

    for csv_file in CSV_FILES:
        path = os.path.join(PREPROC_DIR, csv_file)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)

        real_count = 0
        place_count = 0
        for idx, row in df.iterrows():
            key = str(row.get('Pan-STARRS_objID', f"{row['Pan-STARRS_RAJ2000']}_{row['Pan-STARRS_DEJ2000']}"))
            if key in results and results[key].get('w1flux') is not None:
                df.at[idx, 'w1flux'] = results[key]['w1flux']
                df.at[idx, 'w2flux'] = results[key]['w2flux']
                real_count += 1
            else:
                # Keep placeholder
                place_count += 1

        df.to_csv(path, index=False)
        log(f"  {csv_file}: real={real_count}, placeholder={place_count}")

    log(f"\nDone! Updated CSVs in {PREPROC_DIR}")
    log(f"CatWISE results cached in {CACHE_FILE}")

if __name__ == '__main__':
    main()
