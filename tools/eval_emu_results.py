# -*- coding: utf-8 -*-
"""
EMU 三组对照评估脚本 (eval_emu_results.py)

合并 run_emu_inference.py 三组输出 (raw/uniform/prior), 输出对比报告:
  - 每组: N, Accuracy, Host 召回, 非宿主特异性, GT1/GT0 平均 host_prob 与分离度
  - 按正样本形态分层 (DRAGNs Table7 Tags: FR1/FR2/BT 等) 的分离度 (若形态标签可匹配)
  - 组间对比: prior vs uniform 的分离度增益

用法:
  python tools/eval_emu_results.py [--results-dir crossmatch/training_notes]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "crossmatch", "training_notes")
MODES = ["raw", "uniform", "prior", "xgb"]


def log(msg):
    print(str(msg).encode('ascii', errors='replace').decode('ascii'))


def metrics(df, tag):
    hp = df["Host_prob"].values
    gt = df["Ground_Truth"].values
    pred = df["Prediction"].values
    if len(df) == 0:
        return None
    n_host = int((gt == 1).sum())
    n_non = int((gt == 0).sum())
    tp = int(((gt == 1) & (pred == 1)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    host_recall = tp / n_host if n_host else np.nan
    non_spec = tn / n_non if n_non else np.nan
    mu1 = hp[gt == 1].mean() if n_host else np.nan
    mu0 = hp[gt == 0].mean() if n_non else np.nan
    return {
        "mode": tag, "N": len(df), "n_host": n_host, "n_nonhost": n_non,
        "acc": (pred == gt).mean(), "host_recall": host_recall, "nonhost_spec": non_spec,
        "mu_hostGT1": mu1, "mu_hostGT0": mu0, "separation": (mu1 - mu0) if not np.isnan(mu1) and not np.isnan(mu0) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare the three EMU inference modes")
    ap.add_argument("--results-dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    rows = []
    frames = {}
    for mode in MODES:
        p = os.path.join(args.results_dir, "emu_ps1_results_%s.csv" % mode)
        if not os.path.exists(p):
            log("MISSING: %s" % p)
            continue
        df = pd.read_csv(p)
        frames[mode] = df
        rows.append(metrics(df, mode))

    log("=" * 78)
    log("EMU Model-B 三组对照 (零样本, EMU-PS1 100源)")
    log("=" * 78)
    log("%-9s %4s %5s %5s %9s %9s %9s %10s %10s %10s" %
        ("mode", "N", "host", "non", "acc", "hostRec", "nonSpec",
         "muHost1", "muHost0", "separation"))
    for r in rows:
        if r is None:
            continue
        log("%-9s %4d %5d %5d %9.4f %9.4f %9.4f %10.4f %10.4f %10.4f" % (
            r["mode"], r["N"], r["n_host"], r["n_nonhost"], r["acc"],
            r["host_recall"], r["nonhost_spec"], r["mu_hostGT1"], r["mu_hostGT0"],
            r["separation"]))
    log("  (论文参照: 0.646 vs 0.016, sep=0.630; CDFS失败案例 sep=0.026)")
    if "prior" in frames and "uniform" in frames:
        s_p = rows[-1]["separation"] if rows[-1]["mode"] == "prior" else None
        s_u = [r for r in rows if r["mode"] == "uniform"][0]["separation"]
        log("  prior-uniform 分离度增益: %.4f" % ((s_p - s_u) if s_p is not None else np.nan))

    # ---- 形态分层 (正样本) ----
    tags_csv = os.path.join(PROJECT_ROOT, "data", "catalogs", "emu", "dragns_tags.csv")
    if os.path.exists(tags_csv) and "prior" in frames:
        tags = pd.read_csv(tags_csv)
        df = frames["prior"]
        df = df.merge(tags, left_on="EMU_component_name", right_on="source_id", how="left")
        log("")
        log("按 DRAGN 形态分层 (prior 组, 正样本 host_prob):")
        sub = df[df["Ground_Truth"] == 1]
        if len(sub):
            g = sub.groupby("tag")["Host_prob"].agg(["count", "mean"])
            for tag, row in g.iterrows():
                log("  %-16s n=%3d  mean host_prob=%.4f" % (str(tag), int(row["count"]), row["mean"]))
        else:
            log("  (无匹配形态标签)")

    log("")
    log("结论速读:")
    for r in rows:
        if r is None:
            continue
        verdict = "分离度差(预期内)" if r["separation"] < 0.1 else \
                  ("分离度一般" if r["separation"] < 0.3 else "分离度可用")
        log("  %s: separation=%.4f -> %s" % (r["mode"], r["separation"], verdict))


if __name__ == "__main__":
    main()
