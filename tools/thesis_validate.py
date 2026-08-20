"""
按论文方法验证模型 — 使用论文原始数据分割方式

Model A: SDSS光谱分类源 (GALAXY/QSO/STAR各10, 共30源)
Model B: 论文原始Norris测试结果 (1343条, training_notes已有)

严格遵循论文方法，不改进模型。
"""
import csv, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "crossmatch"))

import torch
import torch.nn.functional as F
from mynetwork import CelestialClassficationNet
from FitsImageFolder import FitsImageFolder


def validate_model_a(device="cuda"):
    """论文方法: Model A 在 SDSS 光谱分类源上验证"""
    if not torch.cuda.is_available():
        device = "cpu"

    print("=" * 60)
    print("MODEL A: SDSS Spectroscopic Sources (Paper Method)")
    print("=" * 60)

    # Load model
    print("Loading Model A...")
    model = CelestialClassficationNet()
    model.load_state_dict(torch.load(
        os.path.join(PROJ, "crossmatch", "models", "opt_classification_model_wts.pt"),
        map_location=device, weights_only=True))
    model.to(device).eval()

    # Load SDSS test data via FitsImageFolder (Session 5 data)
    test_root = os.path.join(PROJ, "source_cutouts", "north", "sdss", "ps")
    dataset = FitsImageFolder(root=test_root)
    from optical_dataset import OptDataSet
    test_data = OptDataSet(dataset)

    print(f"Test samples: {len(test_data)}")
    print(f"Classes: {dataset.classes}")

    # Run inference
    results = []
    for i in range(len(test_data)):
        images, _, wise_mag, _ = test_data[i]  # (img [240,240,5], label, wise [2], pos)
        sample_path = dataset.samples[i][0]
        true_class = dataset.classes[dataset.samples[i][1]]

        # Convert numpy to tensor: [240,240,5] -> [5,240,240]
        inputs = torch.from_numpy(images).permute(2, 0, 1).float().unsqueeze(0).to(device)
        wise = torch.from_numpy(wise_mag).float().unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(inputs, wise)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        results.append({
            "source_id": os.path.basename(sample_path),
            "true_class": true_class,
            "galaxy_prob": float(probs[0]),
            "qso_prob": float(probs[1]),
            "star_prob": float(probs[2]),
            "predicted_class": dataset.classes[int(probs.argmax())],
            "correct": int(dataset.classes[int(probs.argmax())] == true_class),
        })

    # Summary
    ga = [r["galaxy_prob"] for r in results]
    qa = [r["qso_prob"] for r in results]
    sa = [r["star_prob"] for r in results]
    acc = sum(r["correct"] for r in results) / len(results) * 100

    print(f"\n{'='*60}")
    print(f"Model A Results ({len(results)} SDSS spectroscopic sources)")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {acc:.1f}%")
    print(f"Galaxy mean: {np.mean(ga):.4f}  median: {np.median(ga):.4f}  std: {np.std(ga):.4f}")
    print(f"QSO mean:    {np.mean(qa):.4f}  median: {np.median(qa):.4f}  std: {np.std(qa):.4f}")
    print(f"STAR mean:   {np.mean(sa):.4f}  median: {np.median(sa):.4f}  std: {np.std(sa):.4f}")

    # Per-class accuracy
    for cls in ["GALAXY", "QSO", "STAR"]:
        cls_results = [r for r in results if r["true_class"] == cls]
        if cls_results:
            cls_acc = sum(r["correct"] for r in cls_results) / len(cls_results) * 100
            cls_gal = np.mean([r["galaxy_prob"] for r in cls_results])
            cls_qso = np.mean([r["qso_prob"] for r in cls_results])
            cls_star = np.mean([r["star_prob"] for r in cls_results])
            print(f"\n  {cls} ({len(cls_results)} samples):")
            print(f"    Accuracy: {cls_acc:.0f}%")
            print(f"    Mean probs: Galaxy={cls_gal:.4f}  QSO={cls_qso:.4f}  STAR={cls_star:.4f}")

    # Save
    out = os.path.join(PROJ, "crossmatch", "training_notes", "thesis_model_a_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print(f"\nSaved: {out}")

    return results


def validate_model_b():
    """论文方法: Model B 在 Norris2006 射电源上的测试结果 (论文原始)"""
    print(f"\n{'='*60}")
    print("MODEL B: Norris2006 Radio Sources (Paper Original Results)")
    print("=" * 60)

    path = os.path.join(PROJ, "crossmatch", "training_notes",
                        "RGZ_all_negative_Norris_testing_notes.csv")

    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    # Parse columns (handle leading spaces)
    keys = list(rows[0].keys())
    gt_k = next(k for k in keys if 'roud' in k.lower())
    pred_k = next(k for k in keys if 'redict' in k.lower())
    host_k = next(k for k in keys if 'Host' in k and 'Non' not in k)
    gal_k = next(k for k in keys if 'alaxy' in k and 'prob' in k.lower())
    qso_k = next(k for k in keys if 'QSO' in k and 'prob' in k.lower())
    star_k = next(k for k in keys if 'STAR' in k and 'prob' in k.lower())

    tp = tn = fp = fn = 0
    host_probs_gt1 = []
    host_probs_gt0 = []
    gal_probs, qso_probs, star_probs = [], [], []

    for r in rows:
        try:
            gt = int(float(r[gt_k]))
            pred = int(float(r[pred_k]))
            hp = float(r[host_k])

            if gt == 1:
                host_probs_gt1.append(hp)
                tp += (pred == 1); fn += (pred != 1)
            else:
                host_probs_gt0.append(hp)
                tn += (pred == 0); fp += (pred != 0)

            gal_probs.append(float(r[gal_k]))
            qso_probs.append(float(r[qso_k]))
            star_probs.append(float(r[star_k]))
        except (ValueError, KeyError):
            continue

    n = tp + tn + fp + fn
    acc = (tp + tn) / n * 100
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Model B Results ({n} Norris2006 radio sources)")
    print(f"{'='*60}")
    print(f"  Ground Truth: RGZ citizen science (host/non-host)")
    print(f"  Test set: Norris2006 CDFS radio components")
    print(f"  Host:Non-host = {tp+fn}:{tn+fp} (1:{tn+fp//(tp+fn) if (tp+fn)>0 else '?'})")
    print()
    print(f"  Accuracy:              {acc:.1f}%")
    print(f"  Host Recall (TP Rate): {rec:.1f}% ({tp}/{tp+fn})")
    print(f"  Precision:             {prec:.1f}%")
    print(f"  Specificity (TN Rate): {spec:.1f}% ({tn}/{tn+fp})")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print()
    print(f"  Model A input distribution:")
    print(f"    Galaxy: mean={np.mean(gal_probs):.4f}")
    print(f"    QSO:    mean={np.mean(qso_probs):.4f}")
    print(f"    STAR:   mean={np.mean(star_probs):.4f}  std={np.std(star_probs):.4f}")
    print()
    print(f"  Host probability:")
    print(f"    GT=Host:     mean={np.mean(host_probs_gt1):.4f} (should be >0.5)")
    print(f"    GT=Non-host: mean={np.mean(host_probs_gt0):.4f} (should be <0.5)")

    return {"n": n, "acc": acc, "recall": rec, "precision": prec, "specificity": spec,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    # Model A: SDSS sources (paper method)
    results_a = validate_model_a()

    # Model B: Norris sources (paper original results)
    results_b = validate_model_b()

    print(f"\n{'='*60}")
    print("SUMMARY: Paper Method Validation")
    print(f"{'='*60}")
    print(f"  Model A: {sum(r['correct'] for r in results_a)}/{len(results_a)} correct ({sum(r['correct'] for r in results_a)/len(results_a)*100:.1f}%)")
    print(f"  Model B: {results_b['acc']:.1f}% accuracy, {results_b['recall']:.1f}% host recall")


if __name__ == "__main__":
    main()
