# -*- coding: utf-8 -*-
"""
EMU 100源选源脚本 (emu_select_sources.py)

从 Selavy 分量星表 + EMU DRAGNs 目录中选出 100 个测试源:
  - 正样本 50: DRAGNs 中有 CATWISE2020 宿主的源
  - 负样本 50: Selavy 分量中远离任何 DRAGN (默认30") 的分量
输出与 data/catalogs/preprocessed_cat/ 同 schema 的 CSV (ps_dataset.py 可直接复用),
另输出一份合并元数据 CSV (emu_ps1_100_sources.csv) 供切图/推理脚本使用。

用法:
  python tools/emu_select_sources.py
      --selavy data/catalogs/emu/selavy_crossmatched.fits
      --dragns data/catalogs/emu/dragns_catalog.fits
      [--catwise data/catalogs/emu/catwise_match.csv]   # 可选: 真实 W1/W2 mJy
      [--n-host 50] [--n-nonhost 50] [--seed 42]
      [--match-arcsec 30] [--force]

幂等: 输出已存在则跳过 (DATA_REGISTRY 惯例), --force 覆盖。
注意: WISE 星等缺省用占位 619.5 mJy, 影响 <0.008 mag (VALIDATION 会话5), 首轮可接受。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "catalogs", "emu")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "catalogs", "emu_ps1_samples")

# WISE 占位通量 (mJy), 与项目历史一致 (VALIDATION: 占位 vs 真实 <0.008 mag)
WISE_PLACEHOLDER_FLUX = 619.5


def log(msg):
    print(("[emu_select] " + str(msg)).encode('ascii', errors='replace').decode('ascii'))


def find_col(cols, candidates):
    """在候选列名列表里找第一个存在的列 (大小写不敏感)。"""
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_table(path):
    if not os.path.exists(path):
        log("ERROR: file not found: %s" % path)
        sys.exit(1)
    log("Loading %s ..." % path)
    tab = Table.read(path)
    log("  %d rows, columns: %s" % (len(tab), ", ".join(tab.colnames)))
    return tab


def selavy_columns(tab):
    cols = tab.colnames
    name_col = find_col(cols, ["name", "component_name", "comp_name",
                               "Component_name", "isl_name", "Isl_name", "comp"])
    ra_col = find_col(cols, ["ra", "RA", "ra_deg", "RAJ2000", "ra_j2000"])
    dec_col = find_col(cols, ["dec", "DEC", "dec_deg", "DEJ2000", "dec_j2000"])
    missing = [n for n, c in [("name", name_col), ("ra", ra_col), ("dec", dec_col)] if c is None]
    if missing:
        log("ERROR: selavy catalog missing columns %s. Available: %s"
            % (missing, ", ".join(cols)))
        sys.exit(1)
    return name_col, ra_col, dec_col


def parse_catwise_names(col):
    """把 CATWISE 风格名称 (JHHMMSS.ss±DDMMSS.s) 解析为 (ra_deg, dec_deg) 数组。
    解析失败的行返回 NaN。兼容 FITS bytes 列与尾部空格填充。"""
    import re
    n = len(col)
    ra = np.full(n, np.nan)
    dec = np.full(n, np.nan)
    pat = re.compile(r"^J(\d{2})(\d{2})(\d{2}\.\d*)([+-])(\d{2})(\d{2})(\d{2}\.\d*)$")
    for i, s in enumerate(col):
        if s is None:
            continue
        if isinstance(s, (bytes, np.bytes_)):
            s = s.decode("ascii", errors="ignore")
        s = str(s).strip()
        m = pat.match(s)
        if not m:
            continue
        hh, mm, ss, sign, dd, dm, ds = m.groups()
        ra[i] = 15.0 * (int(hh) + int(mm) / 60.0 + float(ss) / 3600.0)
        d = int(dd) + int(dm) / 60.0 + float(ds) / 3600.0
        dec[i] = -d if sign == "-" else d
    return ra, dec


def dragns_columns(tab):
    """DRAGNs 目录列名不固定 (VizieR/论文版本各异), 用候选表探测。
    仅射电位置列必填 (用于识别'哪些源是 DRAGN'); 宿主坐标列可缺省
    (缺省时正样本光学位置回退到 selavy 星表的 WISE/DES 交叉坐标)。"""
    cols = tab.colnames
    radio_ra = find_col(cols, ["ra", "RA", "ra_j2000", "RAJ2000", "dragn_ra",
                               "Dragn_RA", "radio_ra", "Radio_RA", "ra_deg", "RA_deg",
                               "Box RA centroid", "Box_RA_centroid", "Box RA"])
    radio_dec = find_col(cols, ["dec", "DEC", "dec_j2000", "DEJ2000", "dragn_dec",
                                "Dragn_Dec", "radio_dec", "Radio_Dec", "dec_deg", "DEC_deg",
                                "Box Dec centroid", "Box_Dec_centroid", "Box Dec"])
    host_ra = find_col(cols, ["catwise_ra", "CATWISE_RA", "cw_ra", "CW_RA", "wise_ra",
                              "WISE_RA", "host_ra", "Host_RA", "opt_ra", "Opt_RA",
                              "assoc_ra", "Assoc_RA", "ra_opt", "RA_opt", "ra1", "RA1",
                              "gal_ra", "Gal_RA"])
    host_dec = find_col(cols, ["catwise_dec", "CATWISE_DEC", "cw_dec", "CW_DEC", "wise_dec",
                               "WISE_DEC", "host_dec", "Host_Dec", "opt_dec", "Opt_Dec",
                               "assoc_dec", "Assoc_Dec", "dec_opt", "Dec_opt", "de1", "DE1",
                               "gal_dec", "Gal_Dec"])
    # 宿主以名称列给出 (如 CATWISE ID: JHHMMSS.ss±DDMMSS.s) -> 解析为坐标
    host_name_col = find_col(cols, ["CATWISE ID", "CATWISE_ID", "catwise_id", "catwise",
                                    "CW ID", "WISE ID", "WISE_ID", "wise_id", "IR ID",
                                    "Host ID", "host_id"])
    missing = [n for n, c in [("radio_ra", radio_ra), ("radio_dec", radio_dec)] if c is None]
    if missing:
        log("ERROR: DRAGNs catalog missing columns %s. Available: %s"
            % (missing, ", ".join(cols)))
        sys.exit(1)
    if host_ra is None or host_dec is None:
        if host_name_col is not None:
            log("NOTE: DRAGNs catalog host given as name column '%s' -> parsing coords"
                % host_name_col)
        else:
            log("NOTE: DRAGNs catalog has no host-position columns; "
                "positive-sample optical position will use selavy WISE/DES crossmatch")
    return radio_ra, radio_dec, host_ra, host_dec, host_name_col


def load_catwise(path):
    """可选: 本地 CatWISE 交叉表 (ra, dec, w1flux, w2flux in mJy)。"""
    tab = Table.read(path)
    cols = tab.colnames
    ra = find_col(cols, ["ra", "RA"])
    dec = find_col(cols, ["dec", "DEC", "de", "DE"])
    w1 = find_col(cols, ["w1flux", "W1flux", "w1mpro", "W1mpro", "f_w1", "F_W1"])
    w2 = find_col(cols, ["w2flux", "W2flux", "w2mpro", "W2mpro", "f_w2", "F_W2"])
    if ra is None or dec is None or w1 is None or w2 is None:
        log("WARNING: catwise file missing ra/dec/w1flux/w2flux columns; using placeholders")
        return None
    log("CatWISE table loaded: %d rows (cols ra=%s dec=%s w1=%s w2=%s)"
        % (len(tab), ra, dec, w1, w2))
    return pd.DataFrame({"ra": np.asarray(tab[ra], dtype=float),
                         "dec": np.asarray(tab[dec], dtype=float),
                         "w1flux": np.asarray(tab[w1], dtype=float),
                         "w2flux": np.asarray(tab[w2], dtype=float)})


def nearest_catwise(cw_df, ra, dec, radius_arcsec=3.0):
    """到 CatWISE 表做最近邻匹配, 返回 (w1flux, w2flux) 或占位。"""
    if cw_df is None or len(cw_df) == 0:
        return WISE_PLACEHOLDER_FLUX, WISE_PLACEHOLDER_FLUX
    src = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    cat = SkyCoord(ra=cw_df["ra"].values * u.deg, dec=cw_df["dec"].values * u.deg)
    sep = src.separation(cat)
    idx = np.argmin(sep.arcsec)
    if sep[idx].arcsec <= radius_arcsec:
        return float(cw_df.iloc[idx]["w1flux"]), float(cw_df.iloc[idx]["w2flux"])
    return WISE_PLACEHOLDER_FLUX, WISE_PLACEHOLDER_FLUX


def wise_from_selavy(selavy, idx, cw_df, ra, dec):
    """从 selavy 行取 WISE 通量 (星表自带 w1flux/w2flux 优先), 否则外部 CatWISE 表, 再否则占位。"""
    try:
        w1 = float(selavy['w1flux'][idx])
        w2 = float(selavy['w2flux'][idx])
        if np.isfinite(w1) and np.isfinite(w2) and w1 > 0 and w2 > 0:
            return w1, w2
    except (KeyError, TypeError, ValueError):
        pass
    if cw_df is not None:
        return nearest_catwise(cw_df, ra, dec)
    return WISE_PLACEHOLDER_FLUX, WISE_PLACEHOLDER_FLUX


def main():
    ap = argparse.ArgumentParser(description="EMU 100源选源 (50 host + 50 non-host)")
    ap.add_argument("--selavy", default=os.path.join(DATA_ROOT, "selavy_crossmatched.csv"))
    ap.add_argument("--dragns", default=os.path.join(DATA_ROOT, "dragns_catalog.fits"))
    ap.add_argument("--catwise", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n-host", type=int, default=50)
    ap.add_argument("--n-nonhost", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match-arcsec", type=float, default=30.0,
                    help="负样本须远离任何 DRAGN 射电位置的距离 (arcsec)")
    ap.add_argument("--max-wise-sep", type=float, default=5.0,
                    help="负样本 WISE 交叉匹配角距上限 (arcsec); 星表有 wise_sep_deg 列时生效")
    ap.add_argument("--exclude-csv", default=None,
                    help="需排除的源清单 CSV (如测试集 emu_ps1_100_sources.csv, 防训练/测试泄漏)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_p = os.path.join(args.out_dir, "emu_ps1_p_samples.csv")
    out_n = os.path.join(args.out_dir, "emu_ps1_n_samples.csv")
    out_all = os.path.join(args.out_dir, "emu_ps1_100_sources.csv")
    if all(os.path.exists(f) for f in (out_p, out_n, out_all)) and not args.force:
        log("Outputs already exist, skipping (use --force to redo):")
        for f in (out_p, out_n, out_all):
            log("  " + f)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # ---- Load inputs ----
    selavy = load_table(args.selavy)
    s_name, s_ra, s_dec = selavy_columns(selavy)
    dragns = load_table(args.dragns)
    d_rra, d_rdec, d_hra, d_hdec, d_hname = dragns_columns(dragns)
    cw_df = load_catwise(args.catwise) if args.catwise else None

    selavy_ra = np.asarray(selavy[s_ra], dtype=float)
    selavy_dec = np.asarray(selavy[s_dec], dtype=float)
    dragns_rra = np.asarray(dragns[d_rra], dtype=float)
    dragns_rdec = np.asarray(dragns[d_rdec], dtype=float)
    has_host_cols = d_hra is not None and d_hdec is not None
    if has_host_cols:
        dragns_hra = np.asarray(dragns[d_hra], dtype=float)
        dragns_hdec = np.asarray(dragns[d_hdec], dtype=float)
    elif d_hname is not None:
        # 宿主以名称列给出 (如 CATWISE ID) -> 解析坐标
        dragns_hra, dragns_hdec = parse_catwise_names(np.asarray(dragns[d_hname]))
        has_host_cols = True
        log("Parsed host coords from '%s': %d/%d valid"
            % (d_hname, np.isfinite(dragns_hra).sum(), len(dragns)))
    else:
        # 宿主列缺省: 退化为射电位置 (正样本光学位置后续用 selavy WISE/DES 交叉)
        dragns_hra = dragns_rra.copy()
        dragns_hdec = dragns_rdec.copy()
        log("NOTE: using DRAGN radio position as host fallback (no host columns)")

    # ---- Positive: DRAGNs (有宿主坐标则优先, 否则全部) ----
    valid_host = ~(np.isnan(dragns_hra) | np.isnan(dragns_hdec))
    log("DRAGNs with usable position: %d/%d" % (valid_host.sum(), len(dragns)))

    # 排除清单 (防训练/测试泄漏): source_id 前缀 EMU_DRAGN_xxxxx / EMU_PS_*
    excl_p = set()
    excl_n = set()
    if args.exclude_csv and os.path.exists(args.exclude_csv):
        ex = pd.read_csv(args.exclude_csv)
        for sid in ex["source_id"]:
            sid = str(sid)
            if sid.startswith("EMU_DRAGN_"):
                excl_p.add(int(sid[len("EMU_DRAGN_"):]))
            elif sid.startswith("EMU_PS_"):
                excl_n.add(sid.replace("_", " "))  # selavy Name 用空格格式
        log("Excluding %d P (DRAGN idx) and %d N (Selavy) from %s"
            % (len(excl_p), len(excl_n), args.exclude_csv))
    if excl_p:
        valid_host = valid_host & ~np.isin(np.arange(len(dragns)), sorted(excl_p))
    if valid_host.sum() < args.n_host:
        log("ERROR: not enough DRAGNs (%d < %d)" % (valid_host.sum(), args.n_host))
        sys.exit(1)
    pos_idx = rng.choice(np.where(valid_host)[0], size=args.n_host, replace=False)
    log("Positive sample: %d DRAGNs (seed=%d)" % (len(pos_idx), args.seed))

    # ---- Negative: Selavy components far from any DRAGN ----
    dragn_coord = SkyCoord(ra=dragns_rra * u.deg, dec=dragns_rdec * u.deg)
    selavy_coord = SkyCoord(ra=selavy_ra * u.deg, dec=selavy_dec * u.deg)
    # 每个 Selavy 到最近 DRAGN 射电位置的距离 (astropy 在 ~20万源量级可接受; 更大可换 kdtree)
    _, sep_nn, _ = selavy_coord.match_to_catalog_sky(dragn_coord)
    far_mask = sep_nn.arcsec > args.match_arcsec
    # WISE 交叉匹配角距过滤 (星表带 wise_sep_deg 列时生效)
    if "wise_sep_deg" in selavy.colnames:
        wise_sep = np.asarray(selavy["wise_sep_deg"], dtype=float)
        wise_ok = np.isfinite(wise_sep) & (wise_sep * 3600.0 <= args.max_wise_sep)
        far_mask = far_mask & wise_ok
        log("After WISE sep<=%g\": %d candidates" % (args.max_wise_sep, far_mask.sum()))
    if excl_n:
        # 排除清单中的负样本 (按 selavy Name 匹配)
        selavy_names = np.asarray(selavy[s_name], dtype=str)
        far_mask = far_mask & ~np.isin(selavy_names, sorted(excl_n))
        log("After excluding %d test negatives: %d candidates" % (len(excl_n), far_mask.sum()))
    log("Selavy components > %g\" from any DRAGN: %d/%d"
        % (args.match_arcsec, far_mask.sum(), len(selavy)))
    if far_mask.sum() < args.n_nonhost:
        log("ERROR: not enough far components (%d < %d); lower --match-arcsec/--max-wise-sep"
            % (far_mask.sum(), args.n_nonhost))
        sys.exit(1)
    neg_idx = rng.choice(np.where(far_mask)[0], size=args.n_nonhost, replace=False)
    log("Negative sample: %d Selavy components (seed=%d)" % (len(neg_idx), args.seed))

    # 每个 DRAGN 宿主位置到最近 Selavy 分量 (用于取 WISE 通量与光学宿主坐标)
    # astropy 匹配不允许 NaN -> 只对有效宿主匹配, 索引映射回全长数组
    host_valid = np.isfinite(dragns_hra) & np.isfinite(dragns_hdec)
    idx_d2s = np.full(len(dragns), -1, dtype=int)
    sep_d2s = np.full(len(dragns), np.inf)
    if host_valid.any():
        idx_v, sep_v, _ = SkyCoord(ra=dragns_hra[host_valid] * u.deg,
                                   dec=dragns_hdec[host_valid] * u.deg) \
            .match_to_catalog_sky(selavy_coord)
        idx_d2s[host_valid] = idx_v
        sep_d2s[host_valid] = sep_v.arcsec

    # ---- Build rows ----
    schema = ["VLASS_component_name", "Pan-STARRS_RAJ2000", "Pan-STARRS_DEJ2000",
              "w1flux", "w2flux", "PS_class", "Pan-STARRS_objID"]
    pos_rows, neg_rows, meta_rows = [], [], []

    for i in pos_idx:
        # i 是 DRAGNs 表索引: 射电位置用 DRAGN 射电坐标
        radio_ra, radio_dec = float(dragns_rra[i]), float(dragns_rdec[i])
        # 光学宿主位置: DRAGN 宿主列优先; 否则最近 selavy 的 DES/WISE 交叉位置 (<10")
        n_sel = int(idx_d2s[i])
        d2s_sep = sep_d2s[i]
        if has_host_cols and np.isfinite(dragns_hra[i]) and np.isfinite(dragns_hdec[i]):
            host_ra, host_dec = float(dragns_hra[i]), float(dragns_hdec[i])
        elif "des_RA_deg" in selavy.colnames and d2s_sep <= 10.0 and \
                np.isfinite(selavy["des_RA_deg"][n_sel]):
            host_ra = float(selavy["des_RA_deg"][n_sel])
            host_dec = float(selavy["des_DEC_deg"][n_sel])
        elif "wise_ra_deg" in selavy.colnames and d2s_sep <= 10.0 and \
                np.isfinite(selavy["wise_ra_deg"][n_sel]):
            host_ra = float(selavy["wise_ra_deg"][n_sel])
            host_dec = float(selavy["wise_dec_deg"][n_sel])
        else:
            host_ra, host_dec = radio_ra, radio_dec
        # WISE: 取最近 selavy 分量的通量 (匹配角距 <5" 才可信, 否则占位)
        if d2s_sep <= 5.0:
            w1, w2 = wise_from_selavy(selavy, n_sel, cw_df, host_ra, host_dec)
        else:
            w1 = w2 = WISE_PLACEHOLDER_FLUX
        comp = "EMU_DRAGN_%05d" % i
        pos_rows.append([comp, host_ra, host_dec, w1, w2, "P", comp])
        meta_rows.append({"source_id": comp, "radio_RA": radio_ra, "radio_Dec": radio_dec,
                          "opt_RA": host_ra, "opt_Dec": host_dec,
                          "class": "P", "label_source": "DRAGNs"})

    for i in neg_idx:
        # i 是 Selavy 表索引
        radio_ra, radio_dec = selavy_ra[i], selavy_dec[i]
        comp = str(selavy[s_name][i]).replace(" ", "_")
        w1, w2 = wise_from_selavy(selavy, i, cw_df, radio_ra, radio_dec)
        neg_rows.append([comp, radio_ra, radio_dec, w1, w2, "N", comp])
        meta_rows.append({"source_id": comp, "radio_RA": radio_ra, "radio_Dec": radio_dec,
                          "opt_RA": radio_ra, "opt_Dec": radio_dec,
                          "class": "N", "label_source": "Selavy(far)"})

    df_p = pd.DataFrame(pos_rows, columns=schema)
    df_n = pd.DataFrame(neg_rows, columns=schema)
    df_all = pd.DataFrame(meta_rows)

    df_p.to_csv(out_p, index=False)
    df_n.to_csv(out_n, index=False)
    df_all.to_csv(out_all, index=False)
    log("Saved:")
    log("  %s (%d rows, P)" % (out_p, len(df_p)))
    log("  %s (%d rows, N)" % (out_n, len(df_n)))
    log("  %s (%d rows, meta)" % (out_all, len(df_all)))
    log("DONE")


if __name__ == "__main__":
    main()
