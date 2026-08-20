# -*- coding: utf-8 -*-
"""
EMU 射电切图下载脚本 (download_emu_radio_cutouts.py)

从 AS101 derived data products 的 EMU_PS_IMAGE.taylor.0.fits (18" 统一分辨率 Stokes I 拼图)
按选源清单 (emu_select_sources.py 输出的 emu_ps1_100_sources.csv) 裁切中心射电切图,
存为 source_cutouts/emu_ps1/radio/{source_id}.fits (带切图 WCS 头, 供 ps_dataset 复用)。

用法:
  python tools/download_emu_radio_cutouts.py
      --mosaic data/emu/EMU_PS_IMAGE.taylor.0.fits
      --sources data/catalogs/emu_ps1_samples/emu_ps1_100_sources.csv
      [--out-dir source_cutouts/emu_ps1/radio]
      [--field-arcsec 300]      # 默认 5' x 5' 视场
      [--max-sources 100]       # 调试时可用小值
      [--force]

幂等: 已存在的切图跳过, --force 覆盖。
注意: 18" 分辨率下预处理常数 (CenterCrop(216) 等) 仍按 VLASS 标定,
      正式迁移需按 EMU beam 重标定 (README 注意点2) -- 本脚本只负责切图本身。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(("[emu_cutout] " + str(msg)).encode('ascii', errors='replace').decode('ascii'))


def pixscale_arcsec(wcs):
    """返回 WCS 的像素尺度 (arcsec/pixel)。
    注意 astropy 8 的 proj_plane_pixel_scales() 返回 Quantity, 直接乘 3600 会带单位,
    统一转成纯 float (arcsec/pix)。"""
    try:
        scales = wcs.proj_plane_pixel_scales()  # Quantity, deg/pix
        return float(np.abs(scales[0]).value) * 3600.0
    except Exception:
        cd = wcs.wcs.cd
        if cd is not None:
            return abs(float(cd[0, 0])) * 3600.0
        cdelt = wcs.wcs.cdelt
        return abs(float(cdelt[0])) * 3600.0


def main():
    ap = argparse.ArgumentParser(description="Cut EMU radio cutouts from taylor.0 mosaic")
    ap.add_argument("--mosaic", required=True,
                    help="Path to EMU_PS_IMAGE.taylor.0.fits")
    ap.add_argument("--sources", required=True,
                    help="Meta CSV with columns: source_id, radio_RA, radio_Dec")
    ap.add_argument("--out-dir",
                    default=os.path.join(PROJECT_ROOT, "source_cutouts", "emu_ps1", "radio"))
    ap.add_argument("--field-arcsec", type=float, default=432.0,
                    help="Cutout field size in arcsec (default 432 = 216px @ 2\"/px, 匹配 CenterCrop(216))")
    ap.add_argument("--max-sources", type=int, default=None,
                    help="Limit number of sources (debug)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.mosaic):
        log("ERROR: mosaic not found: %s" % args.mosaic)
        sys.exit(1)
    if not os.path.exists(args.sources):
        log("ERROR: sources CSV not found: %s" % args.sources)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.sources)
    if args.max_sources:
        df = df.head(args.max_sources)
    log("Sources to cut: %d (field=%g arcsec)" % (len(df), args.field_arcsec))

    # memmap 打开, 避免整图进内存
    hdul = fits.open(args.mosaic, memmap=True)
    data = hdul[0].data
    header = hdul[0].header
    # 拼图可能为 4D (1,1,Ny,Nx), Cutout2D 需要 2D
    if data.ndim > 2:
        data = data[(0,) * (data.ndim - 2)]
    wcs = WCS(header).celestial
    scale = pixscale_arcsec(wcs)
    size_px = int(round(args.field_arcsec / scale))
    log("Mosaic shape=%s, pixscale=%.2f arcsec/px, cutout=%d px"
        % (data.shape, scale, size_px))

    done, skipped, failed = 0, 0, 0
    for _, row in df.iterrows():
        sid = str(row["source_id"])
        out_path = os.path.join(args.out_dir, sid + ".fits")
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue
        try:
            pos = SkyCoord(ra=float(row["radio_RA"]) * u.deg,
                           dec=float(row["radio_Dec"]) * u.deg, frame="icrs")
            cutout = Cutout2D(data, position=pos, size=size_px, wcs=wcs,
                              mode="trim", fill_value=np.nan)
            hdu = fits.PrimaryHDU(cutout.data.astype(np.float32))
            hdu.header.update(cutout.wcs.to_header())
            hdu.header["ORIG_RA"] = (float(row["radio_RA"]), "cutout center RA")
            hdu.header["ORIG_DEC"] = (float(row["radio_Dec"]), "cutout center Dec")
            hdu.writeto(out_path, overwrite=args.force)
            done += 1
        except Exception as e:
            failed += 1
            log("  FAIL %s: %s" % (sid, str(e)[:120]))

        if (done + skipped + failed) % 20 == 0:
            log("  progress: done=%d skipped=%d failed=%d" % (done, skipped, failed))

    hdul.close()
    log("DONE: saved=%d skipped=%d failed=%d -> %s" % (done, skipped, failed, args.out_dir))
    if failed:
        log("WARNING: %d cutouts failed, check logs above" % failed)


if __name__ == "__main__":
    main()
