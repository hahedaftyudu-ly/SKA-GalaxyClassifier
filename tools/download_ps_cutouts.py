"""
阶段2: 下载PanSTARRS光学FITS切图

对training_notes中的71个VLASS分量，为每个分量的PS候选体下载5波段(grizy)堆叠FITS。
使用panstamps库，以VLASS分量坐标为圆心下载180角秒视场切图。
每个分量目录包含5个FITS文件(按grizy顺序)。

输出: source_cutouts/cdfs/ps/{VLASS_component_name}/ (每个目录5个FITS)
"""

import os
import sys
# Add tools dir to path for dummy readline (panstamps needs it on Windows)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import shutil
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.logger import log as astropy_log

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
TRAINING_NOTES = os.path.join(PROJECT_ROOT, "crossmatch", "training_notes",
                              "RGZ_all_negative_Norris_testing_notes1.csv")
PS_DIR = os.path.join(PROJECT_ROOT, "source_cutouts", "cdfs", "ps")
LOG_FILE = os.path.join(DATA_ROOT, "logs", "download_ps_log.txt")

os.makedirs(PS_DIR, exist_ok=True)

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line.encode('ascii', errors='replace').decode('ascii'))
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def parse_vlass_coords(name: str):
    """Parse RA/Dec from VLASS component name"""
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

def fix_fits_header_shape(fits_path):
    """Fix NAXIS1/NAXIS2 if they don't match actual data shape"""
    try:
        with fits.open(fits_path, mode='update', memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.is_image:
                    data_shape = hdu.data.shape
                    if len(data_shape) >= 2:
                        if hdu.header.get('NAXIS1') != data_shape[-1]:
                            hdu.header['NAXIS1'] = data_shape[-1]
                        if hdu.header.get('NAXIS2') != data_shape[-2]:
                            hdu.header['NAXIS2'] = data_shape[-2]
        return True
    except Exception as e:
        log(f"    WARNING: Could not fix header: {str(e)[:80]}")
        return False

def download_ps_for_component(comp_name: str, ra_deg: float, dec_deg: float, arcsec: int = 180) -> bool:
    """Download 5-band PanSTARRS stack cutouts for a VLASS component"""
    from panstamps.downloader import downloader

    comp_dir = os.path.join(PS_DIR, comp_name)

    # Skip if already has 5 valid FITS
    if os.path.isdir(comp_dir):
        existing = [f for f in os.listdir(comp_dir) if f.endswith('.fits')]
        if len(existing) == 5:
            return True

    os.makedirs(comp_dir, exist_ok=True)

    try:
        fits_paths, _, _ = downloader(
            log=astropy_log,
            fits=True, jpeg=False, color=False,
            ra=ra_deg, dec=dec_deg,
            imageType='stack',
            filterSet='grizy',
            arcsecSize=arcsec
        ).get()

        if not fits_paths or len(fits_paths) != 5:
            log(f"    Got {len(fits_paths) if fits_paths else 0} files, expected 5")
            return False

        # fits_paths are in grizy order from panstamps
        band_names = ['g', 'r', 'i', 'z', 'y']
        for band, src_path in zip(band_names, fits_paths):
            # Rename to include band for clarity
            dst_name = f"stack_{band}_{comp_name}.fits"
            dst_path = os.path.join(comp_dir, dst_name)
            shutil.copy2(src_path, dst_path)
            # Fix header shape if needed
            fix_fits_header_shape(dst_path)

        return True

    except Exception as e:
        log(f"    Panstamps error: {str(e)[:150]}")
        return False

def main():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"PanSTARRS FITS Download Log\n{'='*60}\n")

    # ---- Load training notes ----
    log("Loading training notes...")
    df_notes = pd.read_csv(TRAINING_NOTES, skipinitialspace=True)
    df_notes.columns = [c.strip() for c in df_notes.columns]
    all_components = df_notes['VLASS_component_name'].unique()
    log(f"  Unique VLASS components: {len(all_components)}")

    # ---- Get center coordinates for each component ----
    log("\nComputing center coordinates...")
    comp_coords = {}
    for comp_name in all_components:
        try:
            ra, dec = parse_vlass_coords(comp_name)
            comp_coords[comp_name] = (ra, dec)
        except Exception as e:
            log(f"  Cannot parse {comp_name}: {e}")

    log(f"  Parsed {len(comp_coords)} component coordinates")
    log(f"  Dec range: {min(c[1] for c in comp_coords.values()):.1f} to {max(c[1] for c in comp_coords.values()):.1f}")
    log(f"  RA range: {min(c[0] for c in comp_coords.values()):.1f} to {max(c[0] for c in comp_coords.values()):.1f}")

    # Check PanSTARRS coverage
    min_dec = min(c[1] for c in comp_coords.values())
    if min_dec < -30:
        log(f"  WARNING: Some sources below Dec=-30, PanSTARRS may not cover them!")

    # ---- Download ----
    log(f"\nDownloading PS cutouts (arcsec=180, 5 bands each)...")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, (comp_name, (ra, dec)) in enumerate(comp_coords.items()):
        comp_dir = os.path.join(PS_DIR, comp_name)
        if os.path.isdir(comp_dir):
            existing = [f for f in os.listdir(comp_dir) if f.endswith('.fits')]
            if len(existing) == 5:
                skip_count += 1
                continue

        log(f"  [{i+1}/{len(comp_coords)}] {comp_name} (RA={ra:.4f}, Dec={dec:.4f})")

        if download_ps_for_component(comp_name, ra, dec):
            success_count += 1
            log(f"    OK")
        else:
            fail_count += 1
            log(f"    FAILED")

        # Be nice to PanSTARRS server
        time.sleep(0.5)

        if (i + 1) % 10 == 0:
            log(f"  Progress: {i+1}/{len(comp_coords)} (ok={success_count}, skip={skip_count}, fail={fail_count})")

    # ---- Summary ----
    log(f"\n{'='*60}")
    log(f"Download Complete!")
    log(f"  Success: {success_count}")
    log(f"  Skipped: {skip_count}")
    log(f"  Failed: {fail_count}")
    log(f"  Total: {len(comp_coords)}")
    log(f"\nOutput directory: {PS_DIR}")

    # Check results
    total_dirs = sum(1 for d in os.listdir(PS_DIR) if os.path.isdir(os.path.join(PS_DIR, d)))
    log(f"  Component directories: {total_dirs}")

if __name__ == '__main__':
    main()
