"""
阶段4: Norris2006交叉匹配

将training_notes中的VLASS分量与Norris2006射电源进行位置交叉匹配，
获取形态分类标签(Type 1-9)和AGN/SF分类作为ground truth。

输出: data/norris_crossmatch_results.csv
"""

import os
import re
import time
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
NORRIS_FILE = os.path.join(DATA_ROOT, "catalogs", "norris06_table1.csv")
TRAINING_NOTES = os.path.join(PROJECT_ROOT, "crossmatch", "training_notes",
                              "RGZ_all_negative_Norris_testing_notes1.csv")
OUTPUT_FILE = os.path.join(DATA_ROOT, "norris_crossmatch_results.csv")
LOG_FILE = os.path.join(DATA_ROOT, "logs", "crossmatch_norris_log.txt")

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line.encode('ascii', errors='replace').decode('ascii'))
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def parse_sexagesimal(ra_str, dec_str):
    """Parse sexagesimal RA/Dec to decimal degrees"""
    # RA: "HH MM SS.sss"
    ra_parts = ra_str.strip().split()
    ra_h = float(ra_parts[0])
    ra_m = float(ra_parts[1])
    ra_s = float(ra_parts[2])
    ra_deg = (ra_h + ra_m/60.0 + ra_s/3600.0) * 15.0

    # Dec: "DD MM SS.s" or "-DD MM SS.s"
    dec_parts = dec_str.strip().split()
    dec_sign = -1 if dec_parts[0].startswith('-') else 1
    dec_d = abs(float(dec_parts[0]))
    dec_m = float(dec_parts[1])
    dec_s = float(dec_parts[2])
    dec_deg = dec_sign * (dec_d + dec_m/60.0 + dec_s/3600.0)

    return ra_deg, dec_deg

def parse_vlass_coords(name: str):
    """Parse RA/Dec from VLASS component name to decimal degrees"""
    prefix = 'VLASS1QLCIR J'
    coord_part = name[len(prefix):]
    if '-' in coord_part:
        ra_str, dec_str = coord_part.rsplit('-', 1)
        dec_sign = -1
    elif '+' in coord_part:
        ra_str, dec_str = coord_part.rsplit('+', 1)
        dec_sign = 1
    else:
        raise ValueError(f"Cannot parse: {name}")

    # Pad RA integer part to 6 digits (HHMMSS)
    ra_parts = ra_str.split('.')
    ra_int = ra_parts[0].zfill(6)
    if len(ra_parts) > 1:
        ra_str_fixed = ra_int + '.' + ra_parts[1]
    else:
        ra_str_fixed = ra_int

    ra_h = float(ra_str_fixed[:2])
    ra_m = float(ra_str_fixed[2:4])
    ra_s = float(ra_str_fixed[4:])
    ra_deg = (ra_h + ra_m/60.0 + ra_s/3600.0) * 15.0

    dec_d = float(dec_str[:2])
    dec_m = float(dec_str[2:4]) if len(dec_str) > 2 else 0.0
    dec_s = float(dec_str[4:]) if len(dec_str) > 4 else 0.0
    dec_deg = dec_sign * (dec_d + dec_m/60.0 + dec_s/3600.0)

    return ra_deg, dec_deg

def main():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Norris2006 Crossmatch Log\n{'='*60}\n")

    # ---- Step 1: Load Norris2006 catalog ----
    log("Step 1: Loading Norris2006 catalog...")
    df_norris = pd.read_csv(NORRIS_FILE)
    log(f"  Norris sources: {len(df_norris)}")
    log(f"  Columns: {list(df_norris.columns)}")

    # Parse coordinates
    norris_coords = []
    norris_info = []
    for _, row in df_norris.iterrows():
        try:
            ra, dec = parse_sexagesimal(str(row['RAJ2000']), str(row['DEJ2000']))
            norris_coords.append(SkyCoord(ra, dec, unit='deg', frame='icrs'))
            norris_info.append({
                'SID': row['SID'],
                'CID': row['CID'],
                'Type': row['Type'],
                'Class': row['Class'] if pd.notna(row['Class']) else '',
                'f_Class': row['f_Class'] if pd.notna(row['f_Class']) else '',
                'z': row['z'] if pd.notna(row['z']) else '',
                'F20cm': row['F20cm'],
            })
        except Exception as e:
            log(f"  Parse error for {row['SID']}: {e}")

    log(f"  Parsed {len(norris_coords)} Norris sources")
    log(f"  Norris Dec range: {min(c.dec.deg for c in norris_coords):.1f} to {max(c.dec.deg for c in norris_coords):.1f}")
    log(f"  Norris RA range: {min(c.ra.deg for c in norris_coords):.1f} to {max(c.ra.deg for c in norris_coords):.1f}")

    # Type distribution
    type_counts = df_norris['Type'].value_counts().sort_index()
    log(f"\n  Norris Type distribution:")
    for t, c in type_counts.items():
        log(f"    Type {int(t)}: {c}")

    # ---- Step 2: Load VLASS components from training notes ----
    log(f"\nStep 2: Loading VLASS components from training notes...")
    df_notes = pd.read_csv(TRAINING_NOTES, skipinitialspace=True)
    df_notes.columns = [c.strip() for c in df_notes.columns]
    all_components = df_notes['VLASS_component_name'].unique()
    log(f"  Unique VLASS components: {len(all_components)}")

    # Parse VLASS coordinates
    vlass_coords_list = []
    vlass_names = []
    for comp_name in all_components:
        try:
            ra, dec = parse_vlass_coords(comp_name)
            vlass_coords_list.append(SkyCoord(ra, dec, unit='deg', frame='icrs'))
            vlass_names.append(comp_name)
        except:
            log(f"  Cannot parse: {comp_name}")

    log(f"  Parsed {len(vlass_coords_list)} component coordinates")

    # ---- Step 3: Cross-match ----
    log(f"\nStep 3: Cross-matching (radius=10 arcsec)...")
    match_radius = 10.0  # arcsec

    results = []
    for i, (vcoord, vname) in enumerate(zip(vlass_coords_list, vlass_names)):
        best_sep = 999
        best_idx = -1
        for j, ncoord in enumerate(norris_coords):
            sep = vcoord.separation(ncoord).arcsec
            if sep < match_radius and sep < best_sep:
                best_sep = sep
                best_idx = j

        if best_idx >= 0:
            info = norris_info[best_idx]
            results.append({
                'VLASS_component_name': vname,
                'VLASS_RA': vcoord.ra.deg,
                'VLASS_Dec': vcoord.dec.deg,
                'Norris_SID': info['SID'],
                'Norris_CID': info['CID'],
                'Norris_RA': norris_coords[best_idx].ra.deg,
                'Norris_Dec': norris_coords[best_idx].dec.deg,
                'Separation_arcsec': round(best_sep, 3),
                'Norris_Type': info['Type'],
                'Norris_Class': info['Class'],
                'Norris_f_Class': info['f_Class'],
                'Norris_z': info['z'],
                'Norris_F20cm_mJy': info['F20cm'],
            })
        else:
            results.append({
                'VLASS_component_name': vname,
                'VLASS_RA': vcoord.ra.deg,
                'VLASS_Dec': vcoord.dec.deg,
                'Norris_SID': '',
                'Norris_CID': '',
                'Norris_RA': '',
                'Norris_Dec': '',
                'Separation_arcsec': '',
                'Norris_Type': '',
                'Norris_Class': '',
                'Norris_f_Class': '',
                'Norris_z': '',
                'Norris_F20cm_mJy': '',
            })

    df_results = pd.DataFrame(results)

    matched = sum(1 for r in results if r['Norris_SID'])
    log(f"  Matched: {matched}/{len(results)}")
    log(f"  Median separation: {np.median([r['Separation_arcsec'] for r in results if r['Separation_arcsec']]):.2f} arcsec")

    # Distribution of matched Norris types
    matched_types = [r['Norris_Type'] for r in results if r['Norris_SID']]
    from collections import Counter
    type_dist = Counter(matched_types)
    log(f"\n  Matched Norris Types:")
    for t in sorted(type_dist.keys()):
        log(f"    Type {int(t)}: {type_dist[t]}")

    # ---- Step 4: Also cross-match at PS source level ----
    log(f"\nStep 4: Cross-matching at PS source level...")
    # Get all PS sources from notes
    all_ps_ra = df_notes['PS source ra'].values
    all_ps_dec = df_notes['PS source dec'].values

    ps_matched = 0
    for ra, dec in zip(all_ps_ra, all_ps_dec):
        c = SkyCoord(ra, dec, unit='deg', frame='icrs')
        for j, ncoord in enumerate(norris_coords):
            if c.separation(ncoord).arcsec < match_radius:
                ps_matched += 1
                break

    log(f"  PS sources within {match_radius}\" of Norris: {ps_matched}/{len(all_ps_ra)}")

    # ---- Save results ----
    df_results.to_csv(OUTPUT_FILE, index=False)
    log(f"\nResults saved to: {OUTPUT_FILE}")

    # ---- Summary for README ----
    log(f"\n{'='*60}")
    log(f"CROSSMATCH SUMMARY")
    log(f"  VLASS components: {len(vlass_names)}")
    log(f"  Matched to Norris: {matched} ({100*matched/len(vlass_names):.1f}%)")
    log(f"  Norris catalog size: {len(df_norris)}")
    log(f"  Search radius: {match_radius} arcsec")

if __name__ == '__main__':
    main()
