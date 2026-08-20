# -*- coding: utf-8 -*-
"""
EMU 模型B测试推理脚本 (run_emu_inference.py)

在 EMU 100 源上跑模型B, 支持三种 A 概率注入模式 (EMU_TEST_PLAN.md 步骤4):
  raw     -- 真实跑 Model A (DES 5波段 + WISE), 端到端南天基线
  uniform -- 注入 [1/3, 1/3, 1/3], 中性占位
  prior   -- 注入物理先验 (默认 [0.45, 0.45, 0.10], 射电选源 ~= 非恒星), 可 --prior 覆盖

重要实现细节 (与训练一致):
  - 模型B的 ps_source_class_probs 输入在训练/推理中实际是 Model A 的 LOGITS
    (cross_matching.py L184/L247: 传 ps_model_outputs, 非 softmax)。
    因此手动注入默认用 logits = log(p) (softmax(log(p)) = p), 可选 --prior-as-probs 直接用 p。
  - 预处理常数 (CenterCrop(216), centroid*4 缩放等) 仍按 VLASS 标定,
    EMU 18\" 分辨率下需重标定 -- 本脚本先按原常数跑基线 (EMU_TEST_PLAN.md 步骤3 说明)。

用法:
  python crossmatch/run_emu_inference.py --probs-mode prior [--prior 0.45,0.45,0.10]
      [--probs-mode raw|uniform|prior] [--prior-as-probs]
      [--radio-root source_cutouts/emu_ps1/radio]
      [--opt-root source_cutouts/emu_ps1/optical]
      [--p-csv data/catalogs/emu_ps1_samples/emu_ps1_p_samples.csv]
      [--n-csv data/catalogs/emu_ps1_samples/emu_ps1_n_samples.csv]
      [--out crossmatch/training_notes/emu_ps1_results_{mode}.csv]
      [--max-samples N]

输出: 结果 CSV (每行含 A 概率/注入值、host 概率、预测、真值) + 终端汇总
      (accuracy, host 召回, 非宿主特异性, GT1/GT0 平均 host_prob 分离度)。
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mynetwork import CelestialClassficationNet
from crossmatch.model_crossmatch import RadioOpticalCrossmatchModel
from crossmatch.ps_dataset import FitsImageSet
from utils import remove_nan, optimize_image, normalize, get_sigma_clip, cal_luptitude


class EmuFitsImageSet(FitsImageSet):
    """FitsImageSet 的 EMU 变体: 不打乱顺序, 跳过 VLASS 名称解码 (EMU 名不兼容),
    centroid/geodesic 支持按像素尺度重标定 (EMU 2\"/px -> 光学 0.25\"/px 应为 x8)。
    返回 12-tuple: 基础 9 项 (VLASS_name 为占位 (0,0,0)) + (分量名, opt_RA, opt_Dec)。"""

    def __init__(self, radio_root_dir, opt_root_dir, ps_positive_samples_csv,
                 ps_negative_samples_csv, centroid_scale=4.0):
        super().__init__(radio_root_dir, opt_root_dir, ps_positive_samples_csv,
                         ps_negative_samples_csv)
        self.centroid_scale = centroid_scale
        # 覆写: 保持 CSV 顺序, 使行号可映射到源
        self.df_samples = pd.concat([self.ps_p_samples, self.ps_n_samples],
                                    ignore_index=True)

    def __getitem__(self, idx):
        opt_sample = self.df_samples.iloc[idx]
        radio_component_name = opt_sample["VLASS_component_name"]

        radio_img_dat, centroid_x, centroid_y, geodesic_x, geodesic_y = \
            self.preprocess_radio_image(radio_component_name)

        ps_cutouts = self.create_PS_sample_cutout(opt_sample)
        cube = np.stack([c.data for c in ps_cutouts], axis=2)
        cube = optimize_image(cube)

        w1flux = opt_sample['w1flux']
        w2flux = opt_sample['w2flux']
        m1, m2 = cal_luptitude(w1flux, w2flux)
        wise_mag = np.asarray([m1, m2])

        pos_info = np.asarray(ps_cutouts[0].position_original)
        scale = self.centroid_scale
        centroid_pos = np.asarray((scale * centroid_x, scale * centroid_y))
        geodesic_pos = np.asarray((scale * geodesic_x, scale * geodesic_y))

        label = 1 if opt_sample["PS_class"] == 'P' else 0
        ps_id = opt_sample['Pan-STARRS_objID']
        ps_ra = opt_sample['Pan-STARRS_RAJ2000']
        ps_dec = opt_sample['Pan-STARRS_DEJ2000']
        vlass_name = (0, 0.0, 0.0)  # EMU 名称不兼容 VLASS 解码, 占位 (EMU 推理不使用)

        row = self.df_samples.iloc[idx]
        return (radio_img_dat, cube, wise_mag, pos_info, centroid_pos, geodesic_pos,
                label, (ps_id, ps_ra, ps_dec), vlass_name,
                row["VLASS_component_name"],
                float(row["Pan-STARRS_RAJ2000"]),
                float(row["Pan-STARRS_DEJ2000"]))


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line.encode('ascii', errors='replace').decode('ascii'))


def radio_to_tensor(img, target=216):
    """VLASS 训练时的 radio transform = ToTensor + CenterCrop(216)。
    EMU 图尺寸未知, 小于 216 时上采样补齐, 大于则中心裁剪 (近似, 待重标定)。"""
    t = torch.from_numpy(np.asarray(img, dtype=np.float32))
    if t.ndim == 2:
        t = t[None]
    h, w = t.shape[-2], t.shape[-1]
    if h >= target and w >= target:
        y0, x0 = (h - target) // 2, (w - target) // 2
        t = t[:, y0:y0 + target, x0:x0 + target]
    elif h != target or w != target:
        t = F.interpolate(t[None], size=(target, target), mode='bilinear',
                          align_corners=False)[0]
    return t.unsqueeze(0)  # [1,1,H,W]


def opt_cube_to_tensor(cube):
    """[H,W,5] -> [5,H,W] float tensor (与训练一致: ToTensor 对 [H,W,5] 转置)。"""
    return torch.from_numpy(np.asarray(cube, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)


def parse_prior(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 3 or min(v) <= 0:
        raise ValueError("--prior needs 3 positive floats, got %s" % s)
    v = np.asarray(v)
    return v / v.sum()


def main():
    ap = argparse.ArgumentParser(description="EMU Model-B inference with A-prob injection modes")
    ap.add_argument("--radio-root",
                    default=os.path.join(PROJECT_ROOT, "source_cutouts", "emu_ps1", "radio"))
    ap.add_argument("--opt-root",
                    default=os.path.join(PROJECT_ROOT, "source_cutouts", "emu_ps1", "optical"))
    ap.add_argument("--p-csv",
                    default=os.path.join(PROJECT_ROOT, "data", "catalogs", "emu_ps1_samples",
                                         "emu_ps1_p_samples.csv"))
    ap.add_argument("--n-csv",
                    default=os.path.join(PROJECT_ROOT, "data", "catalogs", "emu_ps1_samples",
                                         "emu_ps1_n_samples.csv"))
    ap.add_argument("--model-b",
                    default=os.path.join(PROJECT_ROOT, "crossmatch", "models",
                                         "RGZ_all_negative_crossmatch_model_wts.pt"))
    ap.add_argument("--model-a",
                    default=os.path.join(PROJECT_ROOT, "crossmatch", "models",
                                         "opt_classification_model_wts.pt"))
    ap.add_argument("--probs-mode",
                    choices=["raw", "uniform", "prior", "xgb"], required=True)
    ap.add_argument("--prior", default="0.45,0.45,0.10",
                    help="prior mode 的概率向量 (默认射电选源~非恒星)")
    ap.add_argument("--prior-as-probs", action="store_true",
                    help="直接把概率向量喂给 B (默认转 logits=log(p), 与训练尺度一致)")
    ap.add_argument("--xgb-model",
                    default=os.path.join(PROJECT_ROOT, "data", "cache", "xgb_model_a.json"),
                    help="XGBoost 模型A (WISE-only v1), xgb mode 用")
    ap.add_argument("--out", default=None,
                    help="结果 CSV 路径 (默认 training_notes/emu_ps1_results_{mode}.csv)")
    ap.add_argument("--max-samples", type=int, default=None, help="调试用: 只跑前 N 个源")
    ap.add_argument("--centroid-scale", type=float, default=4.0,
                    help="radio 像素质心 -> 光学 0.25\"/px 网格的缩放 (VLASS 1\"/px 时=4; "
                         "EMU 2\"/px 时=8, 2/0.25=8)")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log("Device: %s | probs-mode=%s" % (device, args.probs_mode))

    if not os.path.exists(args.p_csv) or not os.path.exists(args.n_csv):
        log("ERROR: sample CSVs missing (run tools/emu_select_sources.py first)")
        return
    if not os.path.exists(args.model_b):
        log("ERROR: model B not found: %s" % args.model_b)
        return

    # ---- Models ----
    pos_dims = RadioOpticalCrossmatchModel.detect_pos_dims(args.model_b)
    radio_model = RadioOpticalCrossmatchModel(pos_dims=pos_dims)
    radio_model.load_state_dict(torch.load(args.model_b, map_location=device, weights_only=True))
    radio_model.eval().to(device)
    log("Model B loaded: %s (pos_dims=%d)" % (args.model_b, pos_dims))

    opt_model = None
    xgb_model = None
    xgb_mags = None
    if args.probs_mode == "raw":
        if not os.path.exists(args.model_a):
            log("ERROR: raw mode needs Model A: %s" % args.model_a)
            return
        opt_model = CelestialClassficationNet()
        opt_model.load_state_dict(torch.load(args.model_a, map_location=device, weights_only=True))
        opt_model.eval().to(device)
        log("Model A loaded (raw mode)")
    elif args.probs_mode == "xgb":
        import xgboost as xgb
        if not os.path.exists(args.xgb_model):
            log("ERROR: xgb mode needs model: %s" % args.xgb_model)
            return
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(args.xgb_model)
        # WISE 星等按数据集顺序对齐: flux(mJy) -> mag (与 emu_convert_catalog 同零点, 精确逆变换)
        # 占位通量 619.5 mJy -> NaN (XGBoost 原生处理缺失)
        dfm = pd.concat([pd.read_csv(args.p_csv), pd.read_csv(args.n_csv)], ignore_index=True)
        F0_W1, F0_W2 = 306.681, 170.663
        xgb_mags = []
        for _, r in dfm.iterrows():
            w1, w2 = float(r["w1flux"]), float(r["w2flux"])
            m1 = -2.5 * np.log10(w1 / F0_W1) if (np.isfinite(w1) and abs(w1 - 619.5) > 1.0) else np.nan
            m2 = -2.5 * np.log10(w2 / F0_W2) if (np.isfinite(w2) and abs(w2 - 619.5) > 1.0) else np.nan
            xgb_mags.append((m1, m2))
        n_mag = sum(1 for a, b in xgb_mags if np.isfinite(a) and np.isfinite(b))
        log("XGBoost A loaded (WISE-only v1), mag coverage: %d/%d"
            % (n_mag, len(xgb_mags)))

    # ---- Dataset ----
    ds = EmuFitsImageSet(args.radio_root, args.opt_root, args.p_csv, args.n_csv,
                         centroid_scale=args.centroid_scale)
    n_total = len(ds)
    if args.max_samples:
        n_total = min(n_total, args.max_samples)
    log("Dataset: %d samples" % n_total)

    out_csv = args.out or os.path.join(PROJECT_ROOT, "crossmatch", "training_notes",
                                       "emu_ps1_results_%s.csv" % args.probs_mode)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ---- Inference ----
    results = []
    skipped = 0
    with torch.no_grad():
        for i in range(n_total):
            try:
                sample = ds[i]
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    log("  skip %d: %s" % (i, str(e)[:120]))
                continue

            (radio_img, ps_cube, wise_mag, cutout_pos, centroid, geodesic,
             label, ps_identity, vlass_name, comp_name, ra, dec) = sample

            radio_t = radio_to_tensor(radio_img).float().to(device)
            opt_t = opt_cube_to_tensor(ps_cube).float().to(device)
            wise_t = torch.from_numpy(np.asarray(wise_mag, dtype=np.float32)) \
                .float().to(device).unsqueeze(0)
            pos_t = torch.from_numpy(np.asarray(cutout_pos, dtype=np.float32)) \
                .float().to(device).unsqueeze(0)
            cent_t = torch.from_numpy(np.asarray(centroid, dtype=np.float32)) \
                .float().to(device).unsqueeze(0)
            geo_t = torch.from_numpy(np.asarray(geodesic, dtype=np.float32)) \
                .float().to(device).unsqueeze(0)

            if args.probs_mode == "raw":
                a_logits = opt_model(opt_t, wise_t)          # [1,3] logits
            elif args.probs_mode == "xgb":
                m1, m2 = xgb_mags[i]
                feat = np.array([[m1, m2, m1 - m2]], dtype=np.float32)
                p = xgb_model.predict_proba(feat)[0].astype(np.float32)
                # 与 eval_xgboost_model_a.py 一致: 直接把概率喂给 B
                a_logits = torch.from_numpy(p).to(device).unsqueeze(0)
            else:
                if args.probs_mode == "uniform":
                    p = np.array([1.0 / 3] * 3)
                else:
                    p = parse_prior(args.prior)
                if args.prior_as_probs:
                    a_logits = torch.from_numpy(p.astype(np.float32)).to(device).unsqueeze(0)
                else:
                    # log(p): softmax(log(p)) = p, 与训练时 logits 尺度一致
                    a_logits = torch.from_numpy(np.log(p).astype(np.float32)) \
                        .to(device).unsqueeze(0)

            if args.probs_mode == "xgb":
                a_probs = a_logits.cpu().numpy()[0]   # 已经是概率, 不再 softmax
            else:
                a_probs = F.softmax(a_logits, dim=1).cpu().numpy()[0]
            b_out = radio_model(radio_t, opt_t, a_logits, pos_t, cent_t, geo_t)
            host_probs = F.softmax(b_out, dim=1).cpu().numpy()[0]
            pred = int(torch.argmax(b_out, dim=1).item())
            gt = int(label)

            results.append({
                "EMU_component_name": comp_name,
                "RA": ra,
                "Dec": dec,
                "probs_mode": args.probs_mode,
                "A_Galaxy_prob": round(float(a_probs[0]), 6),
                "A_QSO_prob": round(float(a_probs[1]), 6),
                "A_STAR_prob": round(float(a_probs[2]), 6),
                "Nonhost_prob": round(float(host_probs[0]), 6),
                "Host_prob": round(float(host_probs[1]), 6),
                "Prediction": pred,
                "Ground_Truth": gt,
            })

            if (i + 1) % 20 == 0:
                log("  progress: %d/%d" % (i + 1, n_total))

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    log("Saved %d rows -> %s (skipped %d)" % (len(df), out_csv, skipped))

    # ---- Summary (论文口径) ----
    if len(df) == 0:
        log("WARNING: no results")
        return
    hp = df["Host_prob"].values
    gt = df["Ground_Truth"].values
    pred = df["Prediction"].values
    acc = (pred == gt).mean()
    tp = ((gt == 1) & (pred == 1)).sum()
    n_host = (gt == 1).sum()
    tn = ((gt == 0) & (pred == 0)).sum()
    n_nonhost = (gt == 0).sum()
    sep = hp[gt == 1].mean() - hp[gt == 0].mean()
    log("=" * 60)
    log("SUMMARY [%s]" % args.probs_mode)
    log("  N=%d (host=%d nonhost=%d)" % (len(df), n_host, n_nonhost))
    log("  Accuracy=%.4f" % acc)
    log("  Host recall=%d/%d=%.4f" % (tp, n_host, tp / n_host if n_host else float('nan')))
    log("  Non-host specificity=%d/%d=%.4f"
        % (tn, n_nonhost, tn / n_nonhost if n_nonhost else float('nan')))
    log("  mean host_prob GT=1: %.4f | GT=0: %.4f | separation=%.4f"
        % (hp[gt == 1].mean(), hp[gt == 0].mean(), sep))
    log("  (论文参照: 0.646 vs 0.016, separation 0.630; CDFS失败: 0.026)")


if __name__ == "__main__":
    main()
