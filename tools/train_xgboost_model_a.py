"""
XGBoost 替代模型A原型 (v1: WISE-only 特征, 完全离线)

目标: 模型A只输出 [GALAXY, QSO, STAR] 三分类概率供模型B使用。
用 XGBoost 在 SDSSxWISE 大星表 (394万行, class_01 光谱标签) 上训练,
替代 CNN 版模型A, 摆脱对切图图像的依赖。

特征 (全部来自 SDSSxWISE_cat.tbl, 无缺失):
  w1mpro, w2mpro          CatWISE 剖面拟合 asinh 星等
  w1w2 = w1mpro - w2mpro  W1-W2 颜色 (QSO 强判别量)
  仅保留星等特征: 评估源 (RGZ宿主) 只有 W1/W2 星等, 无 SNR/流量,
  特征集必须与评估完全一致, 否则评估特征落入训练分布外

标签: class_01 -> 0=GALAXY, 1=QSO, 2=STAR  (与 CNN 版模型A输出顺序一致)

输出:
  data/cache/sdssxwise.pkl       (星表 pickle 缓存, 加速迭代)
  data/cache/xgb_model_a.json    (训练好的模型)
  控制台: 验证集准确率/混淆矩阵/每类召回
"""
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))

TBL_PATH = os.path.join(DATA_ROOT, "catalogs", "SDSS_clean_cat", "SDSSxWISE_cat.tbl")
CACHE_PATH = os.path.join(DATA_ROOT, "cache", "sdssxwise.pkl")
MODEL_PATH = os.path.join(DATA_ROOT, "cache", "xgb_model_a.json")

CLASS_MAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}
FEATURES = ["w1mpro", "w2mpro", "w1w2"]
PER_CLASS = 200_000  # 每类抽样数 (训练规模)
RANDOM_STATE = 42


def load_catalog():
    """加载星表, 有 pickle 缓存则秒读"""
    if os.path.exists(CACHE_PATH):
        print(f"[load] cache: {CACHE_PATH}")
        return pd.read_pickle(CACHE_PATH)
    from astropy.io import ascii
    t0 = time.time()
    tbl = ascii.read(TBL_PATH, format='ipac')
    df = tbl.to_pandas()
    df.to_pickle(CACHE_PATH)
    print(f"[load] parsed {len(df)} rows in {time.time()-t0:.0f}s, cached")
    return df


def build_features(df):
    df = df.dropna(subset=["w1mpro", "w2mpro"]).copy()
    df["w1w2"] = df["w1mpro"] - df["w2mpro"]
    df["label"] = df["class_01"].map(CLASS_MAP)
    return df.dropna(subset=["label"])


def main():
    t0 = time.time()
    print("=" * 60)
    print("XGBOOST MODEL A (WISE-only) TRAINING")
    print("=" * 60)

    df = load_catalog()
    df = build_features(df)
    print(f"[feat] {len(df)} rows with valid features")

    # 分层抽样控制规模 (GALAXY 占比大, 抽到 PER_CLASS)
    # 注意: 不能用 groupby().apply() — pandas 会丢掉分组列 'label'
    sampled = pd.concat([
        g.sample(n=min(PER_CLASS, len(g)), random_state=RANDOM_STATE)
        for _, g in df.groupby("label")
    ])
    print(f"[sample] {len(sampled)} rows (GALAXY/QSO/STAR: "
          f"{int((sampled['label']==0).sum())}/{int((sampled['label']==1).sum())}/"
          f"{int((sampled['label']==2).sum())})")

    X = sampled[FEATURES].values
    y = sampled["label"].values

    # 训练/验证 8:2 分层
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"[split] train {len(X_tr)} / val {len(X_va)}")

    import xgboost as xgb
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)

    # ---- 验证集评估 ----
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    y_pred = model.predict(X_va)
    acc = accuracy_score(y_va, y_pred)
    print(f"\n[val] accuracy: {acc:.4f}  ({time.time()-t0:.0f}s elapsed)")
    print("\nconfusion matrix (rows=真值, cols=预测): [GALAXY QSO STAR]")
    print(confusion_matrix(y_va, y_pred))
    print("\n" + classification_report(y_va, y_pred,
          target_names=["GALAXY", "QSO", "STAR"], digits=4))

    # 校准检查: 预测概率的分布宽度 (模型B吃概率, 校准有影响)
    probs = model.predict_proba(X_va)
    print(f"\n[calib] mean prob per class: "
          f"GALAXY={probs[:,0].mean():.3f} QSO={probs[:,1].mean():.3f} STAR={probs[:,2].mean():.3f}")

    model.save_model(MODEL_PATH)
    print(f"\n[save] model -> {MODEL_PATH}")
    print(f"       total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
