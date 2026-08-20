# -*- coding: utf-8 -*-
"""
EMU 模型B 射电分支微调 (finetune_emu_b.py)

策略 (EMU_TEST_PLAN.md §7.2 策略A):
  - 冻结: cnn_opt (光学5ch ResNet18) -- 南天光学域差异不在本轮范围
  - 固定: A 概率输入 = uniform logits (log(1/3), 与推理 --probs-mode uniform 一致)
  - 训练: cnn_radio (射电1ch) + fc1 + fc2
  - 损失: CrossEntropyLoss (与原训练 cross_matching.py 一致), Adam lr=1e-4 wd=1e-4

数据: data/catalogs/emu_train_samples/ (emu_select_sources.py --exclude-csv 生成, 已排除测试100源)
预处理缓存: source_cutouts/.tmp_emu_train_cache/{comp}.npz (radio+光学+位置, 只算一次)

用法:
  python crossmatch/finetune_emu_b.py
      [--radio-root source_cutouts/emu_ps1/radio]
      [--opt-root source_cutouts/emu_ps1/optical]
      [--p-csv data/catalogs/emu_train_samples/emu_ps1_p_samples.csv]
      [--n-csv data/catalogs/emu_train_samples/emu_ps1_n_samples.csv]
      [--init crossmatch/models/RGZ_all_negative_crossmatch_model_wts.pt]
      [--out crossmatch/models/emu_finetuned_b.pt]
      [--epochs 4] [--batch-size 16] [--lr 1e-4] [--centroid-scale 8]

输出: 微调权重 (仅保存 state_dict, 架构与 pos_dims=3 兼容, 可直接跑 run_emu_inference.py)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from crossmatch.model_crossmatch import RadioOpticalCrossmatchModel
from crossmatch.ps_dataset import FitsImageSet
from utils import optimize_image, cal_luptitude

CACHE_DIR = os.path.join(PROJECT_ROOT, "source_cutouts", ".tmp_emu_train_cache")


def log(msg):
    print(("[finetune] " + str(msg)).encode('ascii', errors='replace').decode('ascii'))


class CachedEmuTrainSet(Dataset):
    """每源重活预处理 (射电骨架/光学叠波段) 只做一次, 缓存到 npz。"""

    def __init__(self, p_csv, n_csv, radio_root, opt_root, centroid_scale=8.0,
                 force_cache=False):
        self.df = pd.concat([pd.read_csv(p_csv), pd.read_csv(n_csv)], ignore_index=True)
        self.radio_root = radio_root
        self.opt_root = opt_root
        self.centroid_scale = centroid_scale
        self.force_cache = force_cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        # 预处理器实例 (仅借用其方法)
        self._fs = FitsImageSet(radio_root, opt_root, p_csv, n_csv)
        self.failed = set()
        self._prepare_all()

    def _prepare_all(self):
        t0 = time.time()
        n_ok = 0
        for idx in range(len(self.df)):
            comp = str(self.df.iloc[idx]["VLASS_component_name"])
            cache_path = os.path.join(CACHE_DIR, comp + ".npz")
            if os.path.exists(cache_path) and not self.force_cache:
                n_ok += 1
                continue
            try:
                self._prepare_one(idx)
                n_ok += 1
            except Exception as e:
                self.failed.add(comp)
                if len(self.failed) <= 5:
                    log("  cache FAIL %s: %s" % (comp, str(e)[:100]))
            if (idx + 1) % 200 == 0:
                log("  cache progress %d/%d (ok=%d fail=%d, %.0fs)"
                    % (idx + 1, len(self.df), n_ok, len(self.failed), time.time() - t0))
        log("Cache ready: %d/%d ok, %d failed (%.0fs)"
            % (n_ok, len(self.df), len(self.failed), time.time() - t0))

    def _prepare_one(self, idx):
        row = self.df.iloc[idx]
        comp = str(row["VLASS_component_name"])
        # 射电预处理 (与推理管线同一函数)
        radio_img, cx, cy, gx, gy = self._fs.preprocess_radio_image(comp)
        # 光学 5 波段 (按文件名排序 = 1_g,2_r,3_i,4_z,5_Y)
        cutouts = self._fs.create_PS_sample_cutout(row)
        cube = np.stack([c.data for c in cutouts], axis=2)
        # NaN 消毒: DES 切图边缘可能含 NaN 像素; 全零平面会让 optimize_image 0/0=NaN
        radio_img = np.nan_to_num(radio_img, nan=0.0)
        cube = np.nan_to_num(cube, nan=0.0)
        cube = optimize_image(cube)
        cube = np.nan_to_num(cube, nan=0.0)   # 全零平面 0/0 兜底
        pos_info = np.asarray(cutouts[0].position_original)
        label = 1 if str(row["PS_class"]) == "P" else 0
        np.savez_compressed(os.path.join(CACHE_DIR, comp + ".npz"),
                            radio=radio_img.astype(np.float32),
                            cube=cube.astype(np.float32),
                            pos=pos_info.astype(np.float32),
                            cx=float(cx), cy=float(cy), gx=float(gx), gy=float(gy),
                            label=int(label))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        comp = str(row["VLASS_component_name"])
        d = np.load(os.path.join(CACHE_DIR, comp + ".npz"))
        s = self.centroid_scale
        radio = torch.nan_to_num(torch.from_numpy(d["radio"])).unsqueeze(0).float()  # [1,H,W]
        cube = torch.nan_to_num(torch.from_numpy(d["cube"])).permute(2, 0, 1).float()  # [5,240,240]
        pos = torch.from_numpy(d["pos"]).float()
        centroid = torch.tensor([s * d["cx"], s * d["cy"]]).float()
        geodesic = torch.tensor([s * d["gx"], s * d["gy"]]).float()
        label = torch.tensor(int(d["label"]))
        return radio, cube, pos, centroid, geodesic, label


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Model-B radio branch on EMU")
    ap.add_argument("--radio-root",
                    default=os.path.join(PROJECT_ROOT, "source_cutouts", "emu_ps1", "radio"))
    ap.add_argument("--opt-root",
                    default=os.path.join(PROJECT_ROOT, "source_cutouts", "emu_ps1", "optical"))
    ap.add_argument("--p-csv",
                    default=os.path.join(PROJECT_ROOT, "data", "catalogs", "emu_train_samples",
                                         "emu_ps1_p_samples.csv"))
    ap.add_argument("--n-csv",
                    default=os.path.join(PROJECT_ROOT, "data", "catalogs", "emu_train_samples",
                                         "emu_ps1_n_samples.csv"))
    ap.add_argument("--init",
                    default=os.path.join(PROJECT_ROOT, "crossmatch", "models",
                                         "RGZ_all_negative_crossmatch_model_wts.pt"))
    ap.add_argument("--out",
                    default=os.path.join(PROJECT_ROOT, "crossmatch", "models",
                                         "emu_finetuned_b.pt"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--centroid-scale", type=float, default=8.0)
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log("Device: %s" % device)

    # ---- Dataset (含缓存构建) ----
    ds = CachedEmuTrainSet(args.p_csv, args.n_csv, args.radio_root, args.opt_root,
                           centroid_scale=args.centroid_scale,
                           force_cache=args.force_cache)
    if ds.failed:
        log("WARNING: %d sources failed preprocessing, will be excluded: %s"
            % (len(ds.failed), ", ".join(list(ds.failed)[:5])))
    valid_idx = [i for i in range(len(ds))
                 if str(ds.df.iloc[i]["VLASS_component_name"]) not in ds.failed]
    if len(valid_idx) < 100:
        log("ERROR: too few valid samples: %d" % len(valid_idx))
        return
    log("Valid samples: %d" % len(valid_idx))

    # 10% 验证集 (分层, 固定 seed)
    rng = np.random.RandomState(42)
    labels = np.array([int(ds.df.iloc[i]["PS_class"] == "P") for i in valid_idx])
    val_mask = np.zeros(len(valid_idx), dtype=bool)
    for lab in [0, 1]:
        idxs = np.where(labels == lab)[0]
        n_val = max(1, int(0.1 * len(idxs)))
        val_mask[rng.choice(idxs, n_val, replace=False)] = True
    train_idx = [valid_idx[i] for i in range(len(valid_idx)) if not val_mask[i]]
    val_idx = [valid_idx[i] for i in range(len(valid_idx)) if val_mask[i]]
    log("Train=%d Val=%d" % (len(train_idx), len(val_idx)))

    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # ---- Model: 加载预训练, 冻结 cnn_opt ----
    model = RadioOpticalCrossmatchModel(pos_dims=3)
    model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    for p in model.cnn_opt.parameters():
        p.requires_grad_(False)
    model.to(device)
    # 冻结的特征提取器置 eval 模式: BN 用运行统计而非批统计, 否则特征随 batch 漂移
    model.cnn_opt.eval()
    n_frozen = sum(1 for p in model.cnn_opt.parameters() if not p.requires_grad)
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    log("Loaded %s | frozen params=%d | trainable params=%d"
        % (args.init, n_frozen, n_train))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)

    # 固定 A 概率输入: uniform logits (与 --probs-mode uniform 一致)
    a_logits = torch.full((1, 3), float(np.log(1.0 / 3)), device=device)

    best_val_acc = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        for radio, cube, pos, cent, geo, label in train_loader:
            radio, cube, pos, cent, geo, label = (radio.to(device), cube.to(device),
                                                  pos.to(device), cent.to(device),
                                                  geo.to(device), label.to(device))
            optimizer.zero_grad()
            a_in = a_logits.expand(radio.size(0), -1)
            outputs = model(radio, cube, a_in, pos, cent, geo)
            loss = criterion(outputs, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            run_loss += loss.item() * radio.size(0)
            run_correct += (torch.argmax(outputs, 1) == label).sum().item()
            run_n += radio.size(0)
        log("Epoch %d/%d train loss=%.4f acc=%.4f (%.0fs)"
            % (epoch + 1, args.epochs, run_loss / run_n, run_correct / run_n,
               time.time() - t0))

        # ---- Val ----
        model.eval()
        v_loss, v_correct, v_n = 0.0, 0, 0
        v_host_prob1, v_host_prob0 = [], []
        with torch.no_grad():
            for radio, cube, pos, cent, geo, label in val_loader:
                radio, cube, pos, cent, geo, label = (radio.to(device), cube.to(device),
                                                      pos.to(device), cent.to(device),
                                                      geo.to(device), label.to(device))
                a_in = a_logits.expand(radio.size(0), -1)
                outputs = model(radio, cube, a_in, pos, cent, geo)
                v_loss += criterion(outputs, label).item() * radio.size(0)
                v_correct += (torch.argmax(outputs, 1) == label).sum().item()
                v_n += radio.size(0)
                hp = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                lab = label.cpu().numpy()
                v_host_prob1.extend(hp[lab == 1])
                v_host_prob0.extend(hp[lab == 0])
        v_acc = v_correct / v_n
        sep = (np.mean(v_host_prob1) - np.mean(v_host_prob0)) if v_host_prob1 and v_host_prob0 else float('nan')
        log("Epoch %d/%d val loss=%.4f acc=%.4f separation=%.4f (host1=%.4f host0=%.4f)"
            % (epoch + 1, args.epochs, v_loss / v_n, v_acc, sep,
               np.mean(v_host_prob1) if v_host_prob1 else float('nan'),
               np.mean(v_host_prob0) if v_host_prob0 else float('nan')))
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), args.out)
            log("  saved best -> %s (val acc=%.4f)" % (args.out, v_acc))

    log("DONE. best val acc=%.4f | weights: %s" % (best_val_acc, args.out))
    log("下一步: python crossmatch/run_emu_inference.py --model-b %s "
        "--probs-mode uniform --centroid-scale 8" % args.out)


if __name__ == "__main__":
    main()
