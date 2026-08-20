"""
按论文第2.1.1节方法, 从 SDSS DR16 specObj 随机抽取 30 个光谱分类源 (GALAXY/QSO/STAR 各10)
并交叉 SDSSxWISE 表(1角秒) 获取 WISE 通量, 输出与 thesis_method_test_sample.csv 同格式。

论文筛选条件 (paper_extract.txt 第43-44页):
  1. 数据源: SDSS DR16 光谱表 specObj
  2. 天区: dec >= -30 (Pan-STARRS 覆盖)
  3. zwarning IN (0, 16)   -- 0=无问题, 16=MANY OUTLIERS(星系高SNR/宽发射线, 可保留)
  4. 与 CatWISE 交叉匹配, 搜索半径 1角秒 最近邻
  5. 随机抽样 GALAXY/QSO/STAR 各5万 (此处仅各10个做验证)

注意: astroquery.sdss 的 query_sql 走旧HTML端点在本机有SSL问题,
     故直接调用 SkyServerWS JSON API (REST, 即"方式二"端点)。

用法: python tools/sample_sdss_30.py [--n 10] [--seed 42] [--out 输出csv]
"""

import os, sys, json, argparse, urllib.parse, time
import numpy as np
import pandas as pd
import requests
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
TBL_PATH = os.path.join(DATA_ROOT, "catalogs", "SDSS_clean_cat", "SDSSxWISE_cat.tbl")
DEFAULT_OUT = os.path.join(DATA_ROOT, "catalogs", "thesis_method_test_sample_api30.csv")

API = "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"


def sdss_query(sql, release="dr16", timeout=120):
    """调用 SkyServerWS JSON API"""
    url = f"https://skyserver.sdss.org/{release}/SkyServerWS/SearchTools/SqlSearch"
    params = {"cmd": sql, "format": "json"}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # 返回 [{TableName, Rows}, ...], 取第一个表
    for block in data:
        if block["TableName"] == "Table1":
            return block["Rows"]
    return []


def sample_class(cls, n=10, retries=3, win=15.0):
    """按论文条件随机抽 n 条某类光谱源。

    注意: 全表 ORDER BY NEWID() 在 500 万行上排序极慢, SkyServerWS 同步接口会超时
    (论文用 CasJobs 批量任务无此限制)。故按 ra 窗口切块, 窗口内随机,
    窗口顺序打乱 → 结果为准随机, 每次请求快(窗口内~20万行)。
    """
    import random as _random
    windows = np.arange(0, 360, win)
    _random.shuffle(windows)
    out = []
    for ra0 in windows:
        if len(out) >= n:
            break
        sql = f"""
        SELECT TOP {n} specObjID, ra, dec, class, z, zwarning, plate, mjd, fiberID
        FROM specObj
        WHERE dec >= -30
          AND zwarning IN (0, 16)
          AND class = '{cls}'
          AND ra BETWEEN {ra0} AND {ra0 + win}
        ORDER BY NEWID()
        """
        for attempt in range(retries):
            try:
                rows = sdss_query(sql, timeout=60)
                if rows:
                    out.extend(rows)
                    break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  [{cls}] ra={ra0:.0f}° 窗口失败: {e}")
                time.sleep(2)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="每类抽样数 (论文为5万, 验证用10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="输出已存在时强制重新生成")
    args = ap.parse_args()

    # 幂等检查 (DATA_REGISTRY.md 惯例 #2): 输出已存在则跳过
    if os.path.exists(args.out) and not args.force:
        print(f"输出已存在: {args.out}")
        print("  跳过抽样 (同一批次不重复获取, 用 --force 强制重新生成)")
        sys.exit(0)

    print("=" * 60)
    print("Step 1: 从 SDSS DR16 specObj 随机抽样 (论文条件)")
    print("=" * 60)

    samples = []
    for cls in ["GALAXY", "QSO", "STAR"]:
        rows = sample_class(cls, args.n)
        print(f"  {cls}: {len(rows)} 条")
        for r in rows:
            samples.append({
                "ra": float(r["ra"]), "dec": float(r["dec"]),
                "class": r["class"], "specObjID": r["specObjID"],
                "z": r["z"], "plate": r["plate"], "mjd": r["mjd"], "fiberID": r["fiberID"],
            })

    if not samples:
        print("!! 未从 SDSS API 取到任何样本")
        sys.exit(1)
    df_api = pd.DataFrame(samples)
    print(f"API 抽样合计: {len(df_api)} 条")

    print("\n" + "=" * 60)
    print("Step 2: 交叉 SDSSxWISE 表 (1角秒最近邻)")
    print("=" * 60)
    if not os.path.exists(TBL_PATH):
        print(f"!! 缺少交叉表: {TBL_PATH}")
        sys.exit(1)
    print("读取 SDSSxWISE 表 (394万行, 约1分钟)...")
    t = Table.read(TBL_PATH, format="ipac")
    print(f"  表行数: {len(t):,}, 列数: {len(t.columns)}")

    # 建立索引: 表内 ra/dec
    cat_coord = SkyCoord(t["ra"].astype(float), t["dec"].astype(float), unit=(u.deg, u.deg))
    src_coord = SkyCoord(df_api["ra"].values, df_api["dec"].values, unit=(u.deg, u.deg))
    idx, d2d, _ = src_coord.match_to_catalog_sky(cat_coord)
    sep_ok = d2d < 1.0 * u.arcsec   # 论文搜索半径 1角秒

    print(f"  1角秒内匹配到: {sep_ok.sum()}/{len(df_api)}")

    # 组装论文格式输出 (对齐 thesis_method_test_sample.csv 全列)
    cols = ["cntr_01", "dist_x", "pang_x", "ra_01", "dec_01", "class_01",
            "source_name", "source_id", "ra", "dec",
            "w1sky", "w1sigsk", "w2sky", "w2sigsk", "w1snr", "w2snr",
            "w1flux", "w1sigflux", "w2flux", "w2sigflux",
            "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro"]
    rows_out = []
    for i, (j, ok) in enumerate(zip(idx, sep_ok)):
        row = df_api.iloc[i]
        if not ok:
            print(f"  !! 源 {row['specObjID']} (ra={row['ra']:.4f}, dec={row['dec']:.4f}) 无1角秒内WISE对应, 跳过")
            continue
        trow = t[j]
        rows_out.append({c: (trow[c] if c in t.colnames else "") for c in cols})

    out_df = pd.DataFrame(rows_out, columns=cols)
    out_df.to_csv(args.out, index=False)
    print(f"\n输出: {args.out}")
    print(f"样本数: {len(out_df)} (GALAXY {sum(out_df['class_01']=='GALAXY')} / "
          f"QSO {sum(out_df['class_01']=='QSO')} / STAR {sum(out_df['class_01']=='STAR')})")


if __name__ == "__main__":
    main()
