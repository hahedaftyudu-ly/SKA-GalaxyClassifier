"""
阶段1: 下载VLASS射电FITS切图

从training_notes获取71个唯一VLASS分量，解析坐标后通过位置匹配
在vlass_cdfs_components.csv中找到对应的CADC URL，下载FITS切图。

输出: source_cutouts/cdfs/radio/{VLASS_component_name}.fits
"""

import os, sys, time, urllib.request, urllib.error
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
VLASS_CSV = os.path.join(DATA_ROOT, "vlass_cdfs_components.csv")
TRAINING_NOTES = os.path.join(PROJECT_ROOT, "crossmatch", "training_notes",
                              "RGZ_all_negative_Norris_testing_notes1.csv")
RADIO_DIR = os.path.join(PROJECT_ROOT, "source_cutouts", "cdfs", "radio")
LOG_FILE = os.path.join(DATA_ROOT, "logs", "download_vlass_log.txt")

os.makedirs(RADIO_DIR, exist_ok=True)

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line.encode('ascii', errors='replace').decode('ascii'))
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def parse_vlass_coords(name: str):
    """Parse RA/Dec (decimal degrees) from VLASS component name.
    Handles missing leading zeros in RA (e.g., J33323.71 -> J033323.71)"""
    prefix = 'VLASS1QLCIR J'
    coord_part = name[len(prefix):]

    if '-' in coord_part:
        ra_part, dec_part = coord_part.rsplit('-', 1)
        dec_sign = -1
    elif '+' in coord_part:
        ra_part, dec_part = coord_part.rsplit('+', 1)
        dec_sign = 1
    else:
        raise ValueError(f"Cannot parse: {name}")

    # Pad RA integer part to 6 digits (HHMMSS)
    ra_parts = ra_part.split('.')
    ra_int = ra_parts[0].zfill(6)
    if len(ra_parts) > 1:
        ra_str = ra_int + '.' + ra_parts[1]
    else:
        ra_str = ra_int

    ra_h = float(ra_str[:2])
    ra_m = float(ra_str[2:4])
    ra_s = float(ra_str[4:])
    ra_deg = (ra_h + ra_m/60.0 + ra_s/3600.0) * 15.0

    dec_d = float(dec_part[:2])
    dec_m = float(dec_part[2:4]) if len(dec_part) > 2 else 0.0
    dec_s = float(dec_part[4:]) if len(dec_part) > 4 else 0.0
    dec_deg = dec_sign * (dec_d + dec_m/60.0 + dec_s/3600.0)

    return ra_deg, dec_deg

def download_fits(url: str, dest_path: str, max_retries=3) -> bool:
    """Download FITS file with retry"""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'GalaxyClassifier/1.0')
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            # Verify valid FITS
            try:
                with fits.open(data, mode='readonly', memmap=False) as hdul:
                    hdul.verify('silentfix')
            except:
                pass  # Some VLASS cutouts have minor header issues
            with open(dest_path, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                log(f"    Failed (attempt {attempt+1}): {str(e)[:100]}")
    return False

def main():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"VLASS Radio FITS Download Log\n{'='*60}\n")

    # Step 1: Load CDFS catalog and build coordinate index
    log("Step 1: Loading VLASS CDFS catalog...")
    df_vlass = pd.read_csv(VLASS_CSV)
    log(f"  Components: {len(df_vlass)}")

    vlass_coords = SkyCoord(df_vlass['RAJ2000'].values, df_vlass['DEJ2000'].values, unit='deg', frame='icrs')
    log(f"  RA range: {df_vlass['RAJ2000'].min():.2f} - {df_vlass['RAJ2000'].max():.2f}")
    log(f"  Dec range: {df_vlass['DEJ2000'].min():.2f} - {df_vlass['DEJ2000'].max():.2f}")

    # Step 2: Load training notes and match
    log("\nStep 2: Matching training notes VLASS components...")
    df_notes = pd.read_csv(TRAINING_NOTES, skipinitialspace=True)
    df_notes.columns = [c.strip() for c in df_notes.columns]
    all_comps = df_notes['VLASS_component_name'].unique()
    log(f"  Unique components: {len(all_comps)}")

    url_map = {}
    no_match = []
    for comp_name in all_comps:
        try:
            ra, dec = parse_vlass_coords(comp_name)
        except Exception as e:
            log(f"  Parse error: {comp_name}: {e}")
            no_match.append(comp_name)
            continue

        c = SkyCoord(ra, dec, unit='deg', frame='icrs')
        sep = c.separation(vlass_coords)
        idx = sep.argmin()
        min_sep = sep[idx].arcsec

        if min_sep < 1.0:
            url = str(df_vlass.iloc[idx]['QLcutout'])
            # Fix: replace ad: with nrao: and strip RUNID (which expires)
            url = url.replace('ad%3AVLASS', 'nrao:VLASS')
            url = url.replace('ad:VLASS', 'nrao:VLASS')
            # Remove RUNID parameter
            import re
            url = re.sub(r'&RUNID%3D[^&]+', '', url)
            url = re.sub(r'&RUNID=[^&]+', '', url)
            matched_name = str(df_vlass.iloc[idx]['CompName'])
            url_map[comp_name] = url
            if min_sep > 0.01:
                log(f"  {comp_name} -> {matched_name} (sep={min_sep:.2f}\")")
        else:
            log(f"  No match: {comp_name} (nearest={min_sep:.1f}\")")
            no_match.append(comp_name)

    log(f"\n  Matched: {len(url_map)}, No match: {len(no_match)}")

    # Step 3: Download
    log(f"\nStep 3: Downloading to {RADIO_DIR}...")
    ok = skip = fail = 0
    total = len(url_map)

    for i, (comp_name, url) in enumerate(url_map.items()):
        dest = os.path.join(RADIO_DIR, f"{comp_name}.fits")
        if os.path.exists(dest):
            skip += 1
            continue

        log(f"  [{i+1}/{total}] {comp_name}")
        if download_fits(url, dest):
            sz = os.path.getsize(dest) / 1024
            ok += 1
            log(f"    OK ({sz:.1f} KB)")
        else:
            fail += 1

        time.sleep(0.25)
        if (i+1) % 10 == 0:
            log(f"  Progress: {i+1}/{total} (ok={ok}, skip={skip}, fail={fail})")

    log(f"\n{'='*60}")
    log(f"COMPLETE: ok={ok}, skip={skip}, fail={fail}, total={total}")
    log(f"Output: {RADIO_DIR}")

if __name__ == '__main__':
    main()
