# -*- coding: utf-8 -*-
"""
EMU Selavy 星表转换脚本 (emu_convert_catalog.py)

把 AS101 derived 集合的 EMU_PS_CATALOG.xml (VOTable, 178921 分量) 转成项目可用的 CSV,
存到 data/catalogs/emu/selavy_crossmatched.csv。

关键处理:
  - 直接利用星表自带的 WISE 星等: w1mag/w2mag -> mJy 通量 (用 utils.cal_luptitude 同一组
    零点和公式反向换算, 保证 WISE 输入与模型训练口径一致):
        flux_mJy = f0 * 10^(-0.4 * mag),  f0_w1=306.681, f0_w2=170.663 (Wright et al. 2010)
  - 保留 DES 交叉匹配坐标 (des_ra/des_dec) 与 5 波段测光, 供光学切图阶段使用。

用法:
  python tools/emu_convert_catalog.py
      [--xml C:/Users/<you>/Downloads/EMU_PS_CATALOG.xml]
      [--out data/catalogs/emu/selavy_crossmatched.csv]
      [--force]
幂等: 输出已存在则跳过。
"""

import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_XML = os.path.join(os.path.expanduser("~"), "Downloads", "EMU_PS_CATALOG.xml")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "catalogs", "emu", "selavy_crossmatched.csv")

# 与 utils.cal_luptitude 同一组零点和 (mJy)
F0_W1 = 306.681
F0_W2 = 170.663


def log(msg):
    print(("[emu_conv] " + str(msg)).encode('ascii', errors='replace').decode('ascii'))


def mag_to_flux(mag, f0):
    """Vega mag -> flux density (mJy): f = f0 * 10^(-0.4 mag)"""
    m = np.asarray(mag, dtype=float)
    return f0 * np.power(10.0, -0.4 * m)


def main():
    ap = argparse.ArgumentParser(description="Convert EMU_PS_CATALOG VOTable to CSV")
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.xml):
        log("ERROR: xml not found: %s" % args.xml)
        sys.exit(1)
    if os.path.exists(args.out) and not args.force:
        log("Output exists, skipping (--force to redo): %s" % args.out)
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()

    log("Parsing VOTable (this may take a few minutes for 255MB)...")
    from astropy.io.votable import parse
    vot = parse(args.xml)
    table = vot.get_first_table()
    data = table.array
    log("Rows: %d | parse time %.1fs" % (len(data), time.time() - t0))

    cols = table.fields  # FIELD definitions (astropy 6+ 用 iter_fields? 兼容两种)
    field_names = [f.name for f in cols]

    def get(name):
        if name not in field_names:
            return None
        col = data[name]
        try:
            return np.asarray(col, dtype=float)
        except (TypeError, ValueError):
            return np.asarray(col, dtype=str)

    # ---- 核心列 ----
    ra = get("ra_deg")
    dec = get("dec_deg")
    name = get("Name")
    comp_id = get("component_id")
    flux_peak = get("flux_peak")
    flux_int = get("flux_int")
    maj = get("maj_axis")
    mina = get("min_axis")
    maj_dc = get("maj_axis_deconv")
    min_dc = get("min_axis_deconv")
    pos_ang_dc = get("pos_ang_deconv")
    spectral_index = get("spectral_index")
    rms_image = get("rms_image")
    wise_name = get("wise_source_name")
    wise_sep = get("wise_separation")
    wise_ra = get("wise_ra")
    wise_dec = get("wise_dec")
    w1mag = get("w1mag")
    w2mag = get("w2mag")
    des_ra = get("des_ra")
    des_dec = get("des_dec")
    des_obj = get("des_coadd_object_id")
    desi_ra = get("desi_RA")
    desi_dec = get("desi_DEC")
    desi_id = get("desi_ID")

    out = {
        "Name": name,
        "component_id": comp_id,
        "RA_deg": ra,
        "DEC_deg": dec,
        "flux_peak_mJy": flux_peak,
        "flux_int_mJy": flux_int,
        "maj_axis_arcsec": maj,
        "min_axis_arcsec": mina,
        "maj_axis_deconv_arcsec": maj_dc,
        "min_axis_deconv_arcsec": min_dc,
        "pos_ang_deconv_deg": pos_ang_dc,
        "spectral_index": spectral_index,
        "rms_image_mJy": rms_image,
        "wise_source_name": wise_name,
        "wise_sep_deg": wise_sep,
        "wise_ra_deg": wise_ra,
        "wise_dec_deg": wise_dec,
    }
    # WISE 通量 (mJy) -- 反向换算, 与模型输入口径一致
    out["w1flux"] = mag_to_flux(w1mag, F0_W1) if w1mag is not None else None
    out["w2flux"] = mag_to_flux(w2mag, F0_W2) if w2mag is not None else None
    out["w1mag"] = w1mag
    out["w2mag"] = w2mag
    out["des_RA_deg"] = des_ra
    out["des_DEC_deg"] = des_dec
    out["des_coadd_object_id"] = des_obj
    out["desi_RA_deg"] = desi_ra
    out["desi_DEC_deg"] = desi_dec
    out["desi_ID"] = desi_id

    import pandas as pd
    df = pd.DataFrame({k: v for k, v in out.items() if v is not None})
    df.to_csv(args.out, index=False)

    n_wise = df["w1flux"].notna().sum()
    n_des = df["des_RA_deg"].notna().sum()
    n_desi = df["desi_RA_deg"].notna().sum()
    log("Saved: %s" % args.out)
    log("  rows=%d | with WISE flux=%d (%.1f%%) | with DES match=%d (%.1f%%) | with DESI=%d"
        % (len(df), n_wise, 100.0 * n_wise / len(df), n_des,
           100.0 * n_des / len(df), n_desi))
    log("  total time %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
