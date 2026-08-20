"""
阶段7: CDFS独立验证 — 生成专业评估报告

输出3种格式:
  1. 控制台打印 — 实时查看
  2. evaluation_report.md — Markdown格式，老师查阅
  3. evaluation_ppt_data.txt — PPT可直接用的关键数字

报告内容: 总体指标 + 混淆矩阵 + Norris2006分层 + 与原始论文对比
"""

import os, sys, json, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_FILE = os.path.join(PROJECT_ROOT, "crossmatch", "training_notes", "cdfs_validation_results.csv")
NORRIS_XM_FILE = os.path.join(PROJECT_ROOT, "data", "norris_crossmatch_results.csv")
MD_REPORT = os.path.join(PROJECT_ROOT, "crossmatch", "training_notes", "evaluation_report.md")
PPT_DATA = os.path.join(PROJECT_ROOT, "crossmatch", "training_notes", "evaluation_ppt_data.txt")

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found. Run run_cdfs_inference.py first!")
        return

    df = pd.read_csv(RESULTS_FILE)
    y_true = df['Ground_Truth'].values
    y_pred = df['Prediction'].values
    total = len(y_true)

    # --- Basic metrics ---
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    correct = tp + tn
    acc = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # --- Norris stratified ---
    norris_strata = []
    if os.path.exists(NORRIS_XM_FILE):
        df_norris = pd.read_csv(NORRIS_XM_FILE)
        df_merged = df.merge(df_norris, on='VLASS_component_name', how='left')

        TYPE_LABELS = {-1:'Unclassified', 1:'AGN/QSO', 2:'Galaxy/ELG', 3:'Starburst',
                       4:'ULIRG', 5:'RadioGal', 6:'Composite', 7:'Spiral', 9:'Unknown'}
        for nt in sorted(df_merged['Norris_Type'].dropna().unique()):
            if nt == '' or pd.isna(nt):
                continue
            s = df_merged[df_merged['Norris_Type'] == nt]
            if len(s) == 0: continue
            sc = int((s['Prediction'] == s['Ground_Truth']).sum())
            norris_strata.append({
                'label': TYPE_LABELS.get(int(nt), f'Type {int(nt)}'),
                'count': len(s), 'correct': sc, 'acc': sc/len(s)
            })

        # Matched vs unmatched
        matched = df_merged[df_merged['Norris_SID'].notna() & (df_merged['Norris_SID'] != '')]
        unmatched = df_merged[~(df_merged['Norris_SID'].notna() & (df_merged['Norris_SID'] != ''))]

    # ---- Build Markdown Report ----
    md = []
    md.append("# CDFS Independent Validation Report")
    md.append(f"\n> Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    md.append(f"> Model: Two-stage CNN (ResNet18 + WISE → Crossmatch)")
    md.append(f"> Test set: Norris2006 VLASS components via CDFS field (N={total})")
    md.append(f"> Environment: Python 3.12, PyTorch 2.6, RTX 4060 4GB")

    md.append("\n---")
    md.append("\n## 1. Overall Performance")
    md.append("\n| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Accuracy | **{acc*100:.1f}%** ({correct}/{total}) |")
    md.append(f"| Precision | {precision*100:.1f}% |")
    md.append(f"| Recall (Sensitivity) | {recall*100:.1f}% |")
    md.append(f"| F1 Score | {f1*100:.1f}% |")

    md.append("\n## 2. Confusion Matrix")
    md.append("\n| | Predicted Non-host | Predicted Host |")
    md.append("|---|---|---|")
    md.append(f"| **Actual Non-host** | {tn} (TN) | {fp} (FP) |")
    md.append(f"| **Actual Host** | {fn} (FN) | {tp} (TP) |")

    md.append(f"\n- True Negative Rate (Specificity): {tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "")
    md.append(f"- True Positive Rate (Recall): {recall*100:.1f}%")

    # Norris stratified
    if norris_strata:
        md.append("\n## 3. Stratified by Norris2006 Morphological Type")
        md.append("\n| Norris Type | Samples | Correct | Accuracy |")
        md.append("|-------------|---------|---------|----------|")
        for s in norris_strata:
            md.append(f"| {s['label']} | {s['count']} | {s['correct']} | {s['acc']*100:.1f}% |")

    # Model A distribution
    if 'Galaxy_prob' in df.columns:
        md.append("\n## 4. Model A (Optical Classification) Output")
        md.append("\n| Class | Mean Probability |")
        md.append("|-------|-----------------|")
        md.append(f"| Galaxy | {df['Galaxy_prob'].mean()*100:.1f}% |")
        md.append(f"| QSO | {df['QSO_prob'].mean()*100:.1f}% |")
        md.append(f"| Star | {df['STAR_prob'].mean()*100:.1f}% |")

    # Data status
    md.append("\n## 5. Data Status")
    md.append(f"\n- Radio (VLASS) FITS: 51/71 components downloaded")
    # Count PS
    ps_dir = os.path.join(PROJECT_ROOT, "source_cutouts", "cdfs", "ps")
    ps_count = sum(1 for d in os.listdir(ps_dir) if os.path.isdir(os.path.join(ps_dir, d))
                   and len([f for f in os.listdir(os.path.join(ps_dir, d)) if f.endswith('.fits')]) == 5)
    md.append(f"- Optical (PanSTARRS) FITS: {ps_count}/71 components with 5-band grizy")
    md.append(f"- WISE magnitudes: real CatWISE values for ~{87} sources, SDSSxWISE median placeholder for rest")
    md.append(f"- Norris2006 crossmatch: 45/71 matched at <10 arcsec")

    # Comparison with original
    md.append("\n## 6. Comparison with Original Paper")
    md.append("\n| Aspect | Original (Lou 2023) | This Validation |")
    md.append("|--------|---------------------|-----------------|")
    md.append(f"| Test set | RGZ + Norris06 (full) | CDFS Norris06 subset |")
    md.append(f"| Samples | ~1343 | {total} |")
    md.append(f"| Accuracy | ~93% (reported) | {acc*100:.1f}% |")
    md.append("| Data | Original FITS (94GB) | Public CADC + PanSTARRS |")
    md.append("| WISE mags | CatWISE full | Partial CatWISE + placeholder |")

    with open(MD_REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"Markdown report: {MD_REPORT}")

    # ---- PPT Data ----
    ppt = []
    ppt.append("=" * 50)
    ppt.append("GALAXYCLASSIFIER CDFS VALIDATION — PPT KEY NUMBERS")
    ppt.append("=" * 50)
    ppt.append(f"\nOverall Accuracy: {acc*100:.1f}%")
    ppt.append(f"Precision: {precision*100:.1f}%")
    ppt.append(f"Recall: {recall*100:.1f}%")
    ppt.append(f"F1: {f1*100:.1f}%")
    ppt.append(f"\nConfusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    ppt.append(f"Total samples: {total}")
    if norris_strata:
        ppt.append(f"\nStratified Accuracy:")
        for s in norris_strata:
            ppt.append(f"  {s['label']}: {s['acc']*100:.1f}% ({s['correct']}/{s['count']})")
    ppt.append(f"\nData: VLASS radio 51/71, PS optical {ps_count}/71")
    ppt.append(f"Norris crossmatch: 45/71 matched")
    ppt.append(f"\n{'='*50}")

    with open(PPT_DATA, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ppt))

    # Print to console
    print("\n" + '\n'.join(ppt))

if __name__ == '__main__':
    main()
