# -*- coding: utf-8 -*-
"""
EMU DES 光学切图下载脚本 (download_des_cutouts.py)

为 EMU 100 源下载 5 波段光学切图, 存为
source_cutouts/emu_ps1/optical/{source_id}/1_g.fits ... 5_Y.fits
(数字前缀保证 FitsImageSet 按文件名排序 = g,r,i,z,Y 通道顺序)

后端说明:
  legacy  (默认): Legacy Surveys 公开切图服务, layer=des-dr1, **仅 g,r,z 三波段**
      -> 通道3(i)与通道5(Y)用 z 波段复制占位 (明确标记为 approx), 用于全管线冒烟测试
  datalab: NOIRLab Astro Data Lab (https://datalab.noirlab.edu 免费注册, 需 token)
      -> 真 5 波段 g,r,i,z,Y, 正式基线用这个

用法:
  python tools/download_des_cutouts.py
      --sources data/catalogs/emu_ps1_samples/emu_ps1_100_sources.csv
      [--backend legacy|datalab] [--token XXX] [--pixscale 0.25]
      [--field-arcsec 60] [--force] [--max-sources N]
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "source_cutouts", "emu_ps1", "optical")

BANDS = ["g", "r", "i", "z", "Y"]
BAND_PREFIX = {b: "%d_%s" % (i + 1, b) for i, b in enumerate(BANDS)}

LEGACY_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
DATALAB_URL = "https://datalab.noirlab.edu/cutout/des_dr2"


def log(msg):
    print(("[des_cutout] " + str(msg)).encode('ascii', errors='replace').decode('ascii'))


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "GalaxyClassifier-EMU-test/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def check_fits(data, tag):
    if len(data) < 2880 or data[:6] != b"SIMPLE":
        raise IOError("%s: not a FITS response (%d bytes, magic=%r)"
                      % (tag, len(data), data[:10]))


def fetch_legacy_grz(ra, dec, pixscale, field_arcsec):
    """Legacy Surveys des-dr1 -> 只支持 g,r,z, 返回单文件 3 面立方体 (g,r,z)。"""
    q = urllib.parse.urlencode({
        "ra": "%.6f" % ra, "dec": "%.6f" % dec,
        "layer": "des-dr1", "pixscale": "%.4f" % pixscale,
        "bands": "grz", "size": str(int(field_arcsec / pixscale)),
    })
    return fetch(LEGACY_URL + "?" + q)


def fetch_datalab_band(ra, dec, pixscale, field_arcsec, band, token):
    q = urllib.parse.urlencode({
        "width_pix": str(int(field_arcsec / pixscale)),
        "height_pix": str(int(field_arcsec / pixscale)),
        "bands": band, "token": token,
    })
    url = "%s/%f/%f?%s" % (DATALAB_URL, ra, dec, q)
    return fetch(url)


def save_plane(fpath, plane, wcs_header_extra, tag):
    from astropy.io import fits as pf
    hdu = pf.PrimaryHDU(np.asarray(plane, dtype=np.float32))
    for k, v in wcs_header_extra.items():
        hdu.header[k] = v
    hdu.header["CHANNEL_NOTE"] = tag
    hdu.writeto(fpath, overwrite=True)


def main():
    ap = argparse.ArgumentParser(description="Download DES 5-band optical cutouts for EMU sources")
    ap.add_argument("--sources", required=True,
                    help="Meta CSV with columns: source_id, opt_RA, opt_Dec")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--backend", choices=["legacy", "datalab"], default="legacy")
    ap.add_argument("--token", default=None, help="Data Lab API token (backend=datalab)")
    ap.add_argument("--pixscale", type=float, default=0.25,
                    help="输出像素尺度 arcsec/px (模型训练用 0.25, 与 PS 一致)")
    ap.add_argument("--field-arcsec", type=float, default=60.0,
                    help="切图视场 (默认 60\" = 240px @ 0.25\"/px, 与训练一致)")
    ap.add_argument("--max-sources", type=int, default=None, help="调试用")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.sources):
        log("ERROR: sources CSV not found: %s" % args.sources)
        sys.exit(1)
    if args.backend == "datalab" and not args.token:
        log("ERROR: backend=datalab needs --token (register free at datalab.noirlab.edu)")
        sys.exit(1)

    df = pd.read_csv(args.sources)
    if args.max_sources:
        df = df.head(args.max_sources)
    os.makedirs(args.out_dir, exist_ok=True)

    size_px = int(args.field_arcsec / args.pixscale)
    log("Sources=%d | backend=%s | %dpx @ %.2f\"/px = %.0f\" field"
        % (len(df), args.backend, size_px, args.pixscale, args.field_arcsec))
    if args.backend == "legacy":
        log("WARNING: legacy des-dr1 只有 g,r,z 三波段; 通道3(i)和5(Y)将用 z 复制占位, "
            "仅供管线冒烟测试. 正式基线请用 datalab 后端. ")

    done, skipped, failed = 0, 0, 0
    failures = []
    for _, row in df.iterrows():
        sid = str(row["source_id"])
        comp_dir = os.path.join(args.out_dir, sid)
        os.makedirs(comp_dir, exist_ok=True)
        fits_files = [os.path.join(comp_dir, BAND_PREFIX[b] + ".fits") for b in BANDS]
        if all(os.path.exists(f) for f in fits_files) and not args.force:
            skipped += 1
            continue
        ra, dec = float(row["opt_RA"]), float(row["opt_Dec"])
        try:
            if args.backend == "legacy":
                data = fetch_legacy_grz(ra, dec, args.pixscale, args.field_arcsec)
                check_fits(data, "legacy")
                import io
                from astropy.io import fits as pf
                with pf.open(io.BytesIO(data)) as h:
                    cube = h[0].data  # (3, Ny, Nx) or (Ny, Nx)
                    hdr = h[0].header
                if cube.ndim == 3:
                    g, r, z = cube[0], cube[1], cube[2]
                else:
                    g = r = z = cube
                wcs_extra = {k: hdr[k] for k in
                             ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1",
                              "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"]
                             if k in hdr}
                save_plane(fits_files[0], g, wcs_extra, "g (real)")
                save_plane(fits_files[1], r, wcs_extra, "r (real)")
                save_plane(fits_files[2], z, wcs_extra, "i APPROX = z (legacy only 3 bands)")
                save_plane(fits_files[3], z, wcs_extra, "z (real)")
                save_plane(fits_files[4], z, wcs_extra, "Y APPROX = z (legacy only 3 bands)")
            else:
                for band, fpath in zip(BANDS, fits_files):
                    if os.path.exists(fpath) and not args.force:
                        continue
                    data = fetch_datalab_band(ra, dec, args.pixscale, args.field_arcsec,
                                              band.lower() if band != "Y" else "Y",
                                              args.token)
                    check_fits(data, band)
                    with open(fpath, "wb") as f:
                        f.write(data)
                    time.sleep(0.3)
            done += 1
        except Exception as e:
            failed += 1
            failures.append("%s: %s" % (sid, str(e)[:120]))
        if (done + skipped + failed) % 10 == 0:
            log("  progress: done=%d skipped=%d failed=%d" % (done, skipped, failed))

    log("DONE: done=%d skipped=%d failed=%d -> %s" % (done, skipped, failed, args.out_dir))
    for f in failures[:10]:
        log("  FAIL %s" % f)
    if failures:
        log("NOTE: legacy 后端仅有 grz; 若需要真 5 波段, 请注册 NOIRLab Data Lab "
            "(datalab.noirlab.edu, 免费) 后重跑: --backend datalab --token <TOKEN>")


if __name__ == "__main__":
    main()
