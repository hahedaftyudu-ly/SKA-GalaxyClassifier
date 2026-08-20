"""
阶段5: 组装推理管线并验证

检查所有文件就位，验证数据完整性，输出就绪报告。
"""

import os, sys, glob
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
REPORT_FILE = os.path.join(DATA_ROOT, "pipeline_readiness_report.txt")

def check(msg, ok):
    status = "OK" if ok else "MISSING"
    print(f"  [{status}] {msg}")
    return ok

def main():
    print("=" * 60)
    print("PIPELINE READINESS CHECK")
    print("=" * 60)

    all_ok = True
    lines = []

    # 1. Model weights
    print("\n--- Model Weights ---")
    weights = [
        "crossmatch/models/opt_classification_model_wts.pt",
        "crossmatch/models/crossmatch_model_wts.pt",
        "crossmatch/models/RGZ_all_negative_crossmatch_model_wts.pt",
    ]
    for w in weights:
        path = os.path.join(PROJECT_ROOT, w)
        ok = os.path.exists(path)
        if ok:
            sz = os.path.getsize(path) / (1024*1024)
            print(f"  [OK] {w} ({sz:.1f} MB)")
        else:
            print(f"  [MISSING] {w}")
            all_ok = False

    # 2. Radio FITS
    print("\n--- Radio FITS ---")
    radio_dir = os.path.join(PROJECT_ROOT, "source_cutouts", "cdfs", "radio")
    radio_files = glob.glob(os.path.join(radio_dir, "*.fits"))
    print(f"  Radio FITS: {len(radio_files)} files")
    if len(radio_files) < 50:
        print(f"  WARNING: Expected 51+, got {len(radio_files)}")
    sizes = [os.path.getsize(f)/1024 for f in radio_files[:5]]
    print(f"  File sizes: ~{np.mean(sizes):.0f} KB each")

    # 3. Optical FITS
    print("\n--- Optical FITS ---")
    ps_dir = os.path.join(PROJECT_ROOT, "source_cutouts", "cdfs", "ps")
    ps_subdirs = [d for d in os.listdir(ps_dir) if os.path.isdir(os.path.join(ps_dir, d))]
    complete = 0
    for d in ps_subdirs:
        fits_count = len(glob.glob(os.path.join(ps_dir, d, "*.fits")))
        if fits_count == 5:
            complete += 1
    print(f"  PS directories: {len(ps_subdirs)}")
    print(f"  Complete (5 FITS each): {complete}")
    if complete < 50:
        print(f"  WARNING: Not all PS cutouts downloaded yet")

    # 4. Preprocessed catalogs
    print("\n--- Preprocessed Catalogs ---")
    csv_dir = os.path.join(DATA_ROOT, "catalogs", "preprocessed_cat")
    csv_files = ["PS_p_Norris06_samples.csv", "PS_n_Norris06_samples.csv",
                 "PS_p_RGZ_samples.csv", "PS_n_samples_RGZ_all.csv"]
    for cf in csv_files:
        path = os.path.join(csv_dir, cf)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Check WISE values
            w1_mean = df['w1flux'].mean()
            w1_is_placeholder = abs(w1_mean - 619.5) < 1.0
            flag = "(PLACEHOLDER)" if w1_is_placeholder else "(REAL CatWISE)"
            print(f"  [OK] {cf}: {len(df)} rows, w1flux mean={w1_mean:.1f} {flag}")
        else:
            print(f"  [MISSING] {cf}")
            all_ok = False

    # 5. Norris crossmatch
    print("\n--- Norris Crossmatch ---")
    norris_file = os.path.join(DATA_ROOT, "catalogs", "norris_crossmatch_results.csv")
    if os.path.exists(norris_file):
        df_n = pd.read_csv(norris_file)
        matched = sum(1 for _, r in df_n.iterrows() if r['Norris_SID'] and str(r['Norris_SID']).strip())
        print(f"  [OK] norris_crossmatch_results.csv: {matched}/{len(df_n)} matched")
    else:
        print(f"  [MISSING] norris_crossmatch_results.csv")
        all_ok = False

    # 6. Python/GPU
    print("\n--- Environment ---")
    import torch
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Summary
    print(f"\n{'='*60}")
    if all_ok and complete >= 50:
        print("READY FOR INFERENCE: python crossmatch/run_cdfs_inference.py")
    else:
        print("NOT READY: Some components still downloading.")
        print(f"  Radio FITS: {len(radio_files)}/51")
        print(f"  PS complete: {complete}/71")
        print(f"  CatWISE: check query_catwise_log.txt")

if __name__ == '__main__':
    main()
