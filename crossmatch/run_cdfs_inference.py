"""
阶段6: CDFS独立验证推理脚本

加载模型A(光学三分类)和模型B(宿主认证)，在Norris06测试集上运行端到端推理。
适配Windows环境(num_workers=0)和正确的权重路径。

输出: crossmatch/training_notes/cdfs_validation_results.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mynetwork import CelestialClassficationNet
from crossmatch.model_crossmatch import RadioOpticalCrossmatchModel
from crossmatch.ps_dataset import FitsImageSet
from crossmatch.xmatch_dataset import XMatchDataset

# ---- Paths ----
VLASS_IMAGE_ROOT = os.path.join(PROJECT_ROOT, "source_cutouts", "cdfs", "radio")
PS_IMAGE_ROOT = os.path.join(PROJECT_ROOT, "source_cutouts", "cdfs", "ps")
MODEL_A_PATH = os.path.join(PROJECT_ROOT, "crossmatch", "models", "opt_classification_model_wts.pt")
# Model B: use base crossmatch model
MODEL_B_PATH = os.path.join(PROJECT_ROOT, "crossmatch", "models", "crossmatch_model_wts.pt")
PS_P_CSV = os.path.join(PROJECT_ROOT, "data", "catalogs", "preprocessed_cat", "filtered_PS_p_Norris06_samples.csv")
PS_N_CSV = os.path.join(PROJECT_ROOT, "data", "catalogs", "preprocessed_cat", "filtered_PS_n_Norris06_samples.csv")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "crossmatch", "training_notes", "cdfs_validation_results.csv")
LOG_FILE = os.path.join(PROJECT_ROOT, "data", "logs", "cdfs_inference_log.txt")

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line.encode('ascii', errors='replace').decode('ascii'))
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def main():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"CDFS Independent Validation Inference Log\n{'='*60}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ---- Step 1: Load models ----
    log("Step 1: Loading models...")

    # Model A: optical classification
    opt_model = CelestialClassficationNet()
    opt_model.load_state_dict(torch.load(MODEL_A_PATH, map_location=device, weights_only=True))
    opt_model.eval().to(device)
    log(f"  Model A loaded: {MODEL_A_PATH}")

    # Model B: crossmatch host identification
    pos_dims = RadioOpticalCrossmatchModel.detect_pos_dims(MODEL_B_PATH)
    radio_model = RadioOpticalCrossmatchModel(pos_dims=pos_dims)
    radio_model.load_state_dict(torch.load(MODEL_B_PATH, map_location=device, weights_only=True))
    radio_model.eval().to(device)
    log(f"  Model B loaded: {MODEL_B_PATH} (pos_dims={pos_dims})")

    # ---- Step 2: Load dataset ----
    log("\nStep 2: Loading test dataset...")

    # Transforms for validation (no augmentation)
    opt_transform = transforms.Compose([transforms.ToTensor()])
    radio_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(216),
    ])

    if not os.path.exists(PS_P_CSV):
        log(f"  ERROR: PS positive CSV not found: {PS_P_CSV}")
        return
    if not os.path.exists(PS_N_CSV):
        log(f"  ERROR: PS negative CSV not found: {PS_N_CSV}")
        return

    testDataset = FitsImageSet(
        radio_root_dir=VLASS_IMAGE_ROOT,
        opt_root_dir=PS_IMAGE_ROOT,
        ps_positive_samples_csv=PS_P_CSV,
        ps_negative_samples_csv=PS_N_CSV,
    )
    testData = XMatchDataset(testDataset,
        radio_transform=radio_transform,
        opt_transform=opt_transform)

    log(f"  Dataset size: {len(testData)}")

    # Filter to only include samples where radio FITS exists
    valid_indices = []
    skipped = 0
    for i in range(len(testData)):
        sample = testDataset[i]
        comp_name = sample[8]  # VLASS_name tuple
        # Check if radio file exists
        # The dataset will fail if radio file doesn't exist, so try-except
        try:
            _ = testData[i]
            valid_indices.append(i)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                log(f"  Skip sample {i}: {str(e)[:100]}")

    if skipped > 0:
        log(f"  Total valid samples: {len(valid_indices)} (skipped {skipped} missing radio/opt)")
        from torch.utils.data import Subset
        testData_filtered = Subset(testData, valid_indices)
    else:
        testData_filtered = testData

    test_loader = DataLoader(testData_filtered, batch_size=1, shuffle=False, num_workers=0)
    log(f"  Test batches: {len(test_loader)}")

    # ---- Step 3: Run inference ----
    log(f"\nStep 3: Running inference...")
    results = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            (radio_img, ps_imgcube, wise_mag, cutout_pos, centroid, geodesic,
             labels, ps_source_id, vlass_name) = batch

            radio_img = radio_img.float().to(device)
            ps_imgcube = ps_imgcube.float().to(device)
            wise_mag = wise_mag.float().to(device)
            cutout_pos = cutout_pos.float().to(device)
            centroid = centroid.float().to(device)
            geodesic = geodesic.float().to(device)

            # Model A: optical classification
            ps_outputs = opt_model(ps_imgcube, wise_mag)
            ps_probs = F.softmax(ps_outputs, dim=1).cpu().numpy()[0]

            # Model B: host identification
            outputs = radio_model(radio_img, ps_imgcube, ps_outputs,
                                  cutout_pos, centroid, geodesic)
            host_probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

            _, predicted = torch.max(outputs.data, 1)

            # Decode VLASS name
            encoded_sign = vlass_name[0].item()
            rra = vlass_name[1].item()
            rdec = vlass_name[2].item()
            sign = '-' if encoded_sign == 0 else '+'
            vlass_comp_name = f'VLASS1QLCIR J{rra}{sign}{rdec}'

            # Get PS source info
            ps_id = ps_source_id[0].item()
            ps_ra = ps_source_id[1].item()
            ps_dec = ps_source_id[2].item()

            results.append({
                'VLASS_component_name': vlass_comp_name,
                'PS_source_id': ps_id,
                'PS_RA': ps_ra,
                'PS_Dec': ps_dec,
                'Galaxy_prob': round(float(ps_probs[0]), 6),
                'QSO_prob': round(float(ps_probs[1]), 6),
                'STAR_prob': round(float(ps_probs[2]), 6),
                'Nonhost_prob': round(float(host_probs[0]), 6),
                'Host_prob': round(float(host_probs[1]), 6),
                'Prediction': int(predicted.item()),
                'Ground_Truth': int(labels.item()),
            })

            if (i+1) % 50 == 0:
                log(f"  Progress: {i+1}/{len(test_loader)}")

    # ---- Step 4: Save results ----
    log(f"\nStep 4: Saving results...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_FILE, index=False)
    log(f"  Saved {len(df_results)} rows to: {RESULTS_FILE}")

    # ---- Quick summary ----
    if len(results) > 0:
        correct = sum(1 for r in results if r['Prediction'] == r['Ground_Truth'])
        acc = correct / len(results)
        log(f"\n{'='*60}")
        log(f"INFERENCE COMPLETE")
        log(f"  Total samples: {len(results)}")
        log(f"  Accuracy: {correct}/{len(results)} = {acc:.4f} ({acc*100:.1f}%)")

        # Per-class stats
        pos_correct = sum(1 for r in results if r['Ground_Truth'] == 1 and r['Prediction'] == 1)
        pos_total = sum(1 for r in results if r['Ground_Truth'] == 1)
        neg_correct = sum(1 for r in results if r['Ground_Truth'] == 0 and r['Prediction'] == 0)
        neg_total = sum(1 for r in results if r['Ground_Truth'] == 0)

        if pos_total > 0:
            log(f"  Host recall: {pos_correct}/{pos_total} = {pos_correct/pos_total:.4f}")
        if neg_total > 0:
            log(f"  Non-host recall: {neg_correct}/{neg_total} = {neg_correct/neg_total:.4f}")
    else:
        log("\n  WARNING: No results! Check that FITS files exist.")

if __name__ == '__main__':
    main()
