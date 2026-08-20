"""
从 SDSS 拉取光学光度 (v2 模型A特征用)

用 SkyServerWS CrossIdSearch 批量查询 PhotoObjAll 的 ugriz 光度:
  psfMag_u/g/r/i/z + modelMag_u/g/r/i/z
交叉认证半径 1 角秒 (与论文一致), 主天体 (nearPrim)。

经验 (2026-08-06 实测):
  - DR16 后端慢且间歇性报错; DR18 后端 ~0.1s/源, 稳定
  - CrossIdSearch 只接受 GET (POST 报缺参数); URL 长度限制在 50-100 行之间
  - 因此用 GET + batch=40 + 4 并发 curl 子进程
  - Windows 下 Python requests 对 skyserver 的 TLS 不稳 (OpenSSL EOF),
    curl (Schannel) 稳定, 故用 curl 子进程

用法:
  python tools/fetch_sdss_photometry.py --n-per-class 30000 --test   # 试跑
  python tools/fetch_sdss_photometry.py --n-per-class 30000           # 正式 (训练集)
  python tools/fetch_sdss_photometry.py --coords-file <csv> --out <pkl>  # 任意坐标 (评估源)
输出:
  data/cache/sdss_phot_train.pkl
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))

TBL_CACHE = os.path.join(DATA_ROOT, "cache", "sdssxwise.pkl")
URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/CrossIdSearch"
RADIUS_ARCSEC = 1.0  # 训练源=SDSS选源, 1" 够; 评估源(PanSTARRS/FIRST坐标)需 3"
BATCH = 40
WORKERS = 3  # 温和并发, 避免打挂共享服务器
CLASS_MAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}

UQUERY = """SELECT u.up_id, u.up_col0 AS col0, p.objID, p.ra, p.dec,
       p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z,
       p.modelMag_u, p.modelMag_g, p.modelMag_r, p.modelMag_i, p.modelMag_z
FROM #x x, #upload u, PhotoTag p
WHERE u.up_id = x.up_id AND x.objID = p.objID
ORDER BY x.up_id"""


def curl_query(paste_text, radius):
    """GET CrossIdSearch, 返回 Rows 列表 (重试4次, 退避)"""
    for attempt in range(4):
        try:
            proc = subprocess.run(
                ["curl", "-s", "--max-time", "120", "-G", URL,
                 "--data-urlencode", "searchtool=CrossID",
                 "--data-urlencode", "searchType=photo",
                 "--data-urlencode", "photoScope=nearPrim",
                 "--data-urlencode", "photoUpType=ra-dec",
                 "--data-urlencode", f"radius={radius}",
                 "--data-urlencode", "firstcol=1",
                 "--data-urlencode", f"paste={paste_text}",
                 "--data-urlencode", f"uquery={UQUERY}",
                 "--data-urlencode", "format=json"],
                capture_output=True, text=True, timeout=150)
            if proc.returncode != 0:
                raise RuntimeError(f"curl rc={proc.returncode}")
            data = json.loads(proc.stdout)
            for table in data:
                if table["TableName"] != "SqlQuery":
                    return table["Rows"]
        except Exception:
            time.sleep(3 * (attempt + 1))  # 3/6/9/12s 退避
    return []


def fetch(coords, out_path, label="", radius=1.0, resume=False,
          max_batches=0):
    """coords: (N,2) 数组; 结果按全局索引 xid 关联; resume=断点续拉;
    max_batches>0 时最多拉 max_batches 批 (涓流模式)"""
    t0 = time.time()
    total_batches = (len(coords) + BATCH - 1) // BATCH
    if max_batches > 0:
        total_batches = min(total_batches, max_batches)

    done_ids = set()
    all_rows = []
    if resume and os.path.exists(out_path):
        prev = pd.read_pickle(out_path)
        if "xid" in prev.columns:
            done_ids = {int(x[1:]) for x in prev["xid"]}
        all_rows = [r for _, r in prev.iterrows()]
        print(f"[{label}] resume: {len(done_ids)} sources already fetched")

    def one_batch(b):
        chunk = coords[b * BATCH:(b + 1) * BATCH]
        g0 = b * BATCH
        if done_ids and all((g0 + i) in done_ids for i in range(len(chunk))):
            return []
        paste_text = "\r\n".join(
            f"X{g0 + i} {r:.6f} {d:.6f}" for i, (r, d) in enumerate(chunk))
        time.sleep(2.0)  # 温和节奏, 避免打挂共享服务器
        return curl_query(paste_text, radius)

    done = len(done_ids)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(one_batch, b): b for b in range(total_batches)}
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)
            done += BATCH
            if done % 200 == 0 or done >= total_batches * BATCH:
                _save(all_rows, out_path)
                print(f"[{label}] {done}/{len(coords)} queried, "
                      f"{len(all_rows)} matched, {time.time()-t0:.0f}s")

    phot = _save(all_rows, out_path)
    print(f"[{label}] {len(phot)} matched / {len(coords)} queried "
          f"({len(phot) / len(coords) * 100:.1f}%), saved -> {out_path}")
    return phot


def _save(rows, out_path):
    phot = pd.DataFrame(rows)
    if not phot.empty:
        phot = phot.rename(columns={"col0": "xid"})
        phot["xid"] = phot["xid"].astype(str)
        phot["ra"] = phot["ra"].astype(float)
        phot["dec"] = phot["dec"].astype(float)
        phot.to_pickle(out_path)
    return phot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=30000)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--coords-file", help="任意坐标 CSV (ra,dec 两列), 评估源用")
    parser.add_argument("--out", default=None, help="coords-file 模式的输出路径")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="交叉认证半径角秒 (评估源非SDSS框架, 建议 3.0)")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="单次最多拉取的批次数 (0=全部; 涓流模式用)")
    args = parser.parse_args()

    if args.coords_file:
        dfc = pd.read_csv(args.coords_file)
        coords = dfc[["ra", "dec"]].values
        out = args.out or os.path.join(DATA_ROOT, "cache", "sdss_phot_eval.pkl")
        fetch(coords, out, label="eval", radius=args.radius,
              max_batches=args.max_batches)
        return

    df = pd.read_pickle(TBL_CACHE)
    df = df.dropna(subset=["w1mpro", "w2mpro"]).copy()
    df["w1w2"] = df["w1mpro"] - df["w2mpro"]
    df["label"] = df["class_01"].map(CLASS_MAP)
    df = df.dropna(subset=["label"])

    n = 3 if args.test else args.n_per_class
    sampled = pd.concat([
        g.sample(n=min(n, len(g)), random_state=42)
        for _, g in df.groupby("label")
    ])
    print(f"[sample] {len(sampled)} sources "
          f"({int((sampled['label'] == 0).sum())}G/"
          f"{int((sampled['label'] == 1).sum())}Q/"
          f"{int((sampled['label'] == 2).sum())}S)")
    # 保存抽样名单 (供训练脚本关联)
    sampled[["ra", "dec", "label"]].to_pickle(
        os.path.join(DATA_ROOT, "cache", "xgb_v2_sample.pkl"))

    coords = sampled[["ra", "dec"]].values
    fetch(coords, os.path.join(DATA_ROOT, "cache", "sdss_phot_train.pkl"),
          label="train", resume=True, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
