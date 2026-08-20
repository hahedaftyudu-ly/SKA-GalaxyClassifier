"""
XGBoost 模型A v2: WISE + SDSS 光学特征

在 v1 (WISE-only) 基础上加入 SDSS ugriz 光度, 目标拆掉 GALAXY<->STAR 混淆:
  WISE:  w1mpro, w2mpro, w1-w2            (3)
  光学:  psfMag_u/g/r/i/z                 (5)
         颜色 u-g, g-r, r-i, i-z          (4)
         形态代理 psfMag_g - modelMag_g    (1)
标签: GALAXY/QSO/STAR (0/1/2), 与 CNN 版模型A输出顺序一致

数据: data/cache/sdssxwise.pkl (WISE+标签)
      data/cache/xgb_v2_sample.pkl (抽样名单, fetch 脚本生成)
      data/cache/sdss_phot_train.pkl (SDSS 光度, fetch 脚本涓流拉取)
训练: 有光学特征匹配的源 (干净集), 超参与 v1 相同
对比: 同一样本上同时训练 WISE-only 基线, 量化光学特征增益
"""
import os
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))

TBL_CACHE = os.path.join(DATA_ROOT, "cache", "sdssxwise.pkl")
SAMPLE_PATH = os.path.join(DATA_ROOT, "cache", "xgb_v2_sample.pkl")
PHOT_PATH = os.path.join(DATA_ROOT, "cache", "sdss_phot_train.pkl")
MODEL_PATH = os.path.join(DATA_ROOT, "cache", "xgb_model_a_v2.json")

WISE_FEATURES = ["w1mpro", "w2mpro", "w1w2"]
OPT_FEATURES = ["psfMag_u", "psfMag_g", "psfMag_r", "psfMag_i", "psfMag_z",
                "ug", "gr", "ri", "iz", "psfm_g"]
RANDOM_STATE = 42


def build_optical(df):
    """从 psfMag/modelMag 构造颜色与形态特征 (NaN 保留, XGBoost 原生处理)"""
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        df[f"{a}{b}"] = df[f"psfMag_{a}"] - df[f"psfMag_{b}"]
    df["psfm_g"] = df["psfMag_g"] - df["modelMag_g"]
    return df


def main():
    t0 = time.time()
    print("=" * 60)
    print("XGBOOST MODEL A v2 (WISE + SDSS optical)")
    print("=" * 60)

    if not os.path.exists(PHOT_PATH):
        print(f"[ABORT] photometry cache not ready: {PHOT_PATH}")
        print("  fetch_sdss_photometry.py 涓流拉取仍在进行, 数据到位后再跑")
        return

    tbl = pd.read_pickle(TBL_CACHE)
    tbl = tbl.dropna(subset=["w1mpro", "w2mpro"]).copy()
    tbl["w1w2"] = tbl["w1mpro"] - tbl["w2mpro"]

    sample = pd.read_pickle(SAMPLE_PATH)
    phot = pd.read_pickle(PHOT_PATH)
    phot["xid_idx"] = phot["xid"].str[1:].astype(int)
    print(f"[data] sample {len(sample)} rows, phot matched {len(phot)}")

    # phot 的 xid 对应 sample 的位置索引
    sub = sample.iloc[phot["xid_idx"].values].copy()
    for c in ["psfMag_u", "psfMag_g", "psfMag_r", "psfMag_i", "psfMag_z",
              "modelMag_g"]:
        sub[c] = phot[c].values
    sub = build_optical(sub)
    sub["label"] = sub["label"].astype(int)
    # WISE 特征从 tbl 按 ra/dec 精确合并 (同一来源行)
    sub = sub.merge(tbl[["ra", "dec", "w1mpro", "w2mpro", "w1w2"]],
                    on=["ra", "dec"], how="left")
    sub = sub.dropna(subset=["w1mpro", "w2mpro"])
    print(f"[data] matched with WISE+optical: {len(sub)} "
          f"({int((sub['label']==0).sum())}G/{int((sub['label']==1).sum())}Q/"
          f"{int((sub['label']==2).sum())}S)")

    from sklearn.model_selection import train_test_split
    X_o = sub[OPT_FEATURES + WISE_FEATURES].values
    X_w = sub[WISE_FEATURES].values
    y = sub["label"].values
    (Xo_tr, Xo_va, Xw_tr, Xw_va, y_tr, y_va) = train_test_split(
        X_o, X_w, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    import xgboost as xgb
    from sklearn.metrics import accuracy_score, confusion_matrix

    results = {}
    for name, Xtr, Xva in [("v2_optical+WISE", Xo_tr, Xo_va),
                           ("v1_wise_only", Xw_tr, Xw_va)]:
        m = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, max_depth=6,
            learning_rate=0.1, n_estimators=500, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist",
            random_state=RANDOM_STATE, eval_metric="mlogloss",
            early_stopping_rounds=20)
        m.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
        acc = accuracy_score(y_va, m.predict(Xva))
        cm = confusion_matrix(y_va, m.predict(Xva))
        results[name] = (acc, cm)
        print(f"\n[{name}] val accuracy: {acc:.4f}  ({time.time()-t0:.0f}s)")
        print("  confusion (rows=truth, cols=pred) [GALAXY QSO STAR]:")
        print("  ", cm.tolist())

    acc_v2, _ = results["v2_optical+WISE"]
    acc_v1, _ = results["v1_wise_only"]
    print(f"\n{'='*60}")
    print(f"OPTICAL GAIN: v1={acc_v1:.4f} -> v2={acc_v2:.4f} "
          f"(+{acc_v2-acc_v1:.4f})")
    print(f"{'='*60}")

    # 保存 v2 模型
    m_v2 = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, max_depth=6,
        learning_rate=0.1, n_estimators=500, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist",
        random_state=RANDOM_STATE, eval_metric="mlogloss",
        early_stopping_rounds=20)
    m_v2.fit(X_o, y)
    m_v2.save_model(MODEL_PATH)
    print(f"[save] v2 model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
