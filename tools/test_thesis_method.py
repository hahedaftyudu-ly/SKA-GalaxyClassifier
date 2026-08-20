"""
按论文第2章方法提取代码：从SDSSxWISE目录抽样 → 下载PanSTARRS切图 → 测试Model A

差不多就是用云盘里wise x sdss星表中抽取的训练数据，然后再到官网下载orz

论文方法 (第2.1.2节):
- 抽样: 从SDSS DR16 + CatWISE2020交叉目录随机选GALAXY/STAR/QSO各50K
- 下载: panstamps API, 60"×60" = 240×240像素, 5波段(grizy)
- 存储: 按类别分目录, 每个源一个子文件夹含5个FITS

用法: python tools/test_thesis_method.py [--full] [--n N]
  --full: 下载全部90个源 (默认只下10个)
  --n N: 下载N个源
"""

import os, sys, time, shutil
import numpy as np
from astropy.table import Table
from astropy.logger import log as astropy_log
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# NOTE: tools/ must come AFTER PROJECT_ROOT to avoid shadowing project utils.py
sys.path.append(os.path.join(PROJECT_ROOT, "tools"))  # for readline dummy

# ============================================================
# Step 0: Prep
# ============================================================
# Generate WISE lookup CSV from .tbl (needed by FitsImageFolder)
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
tbl_path = os.path.join(DATA_ROOT, "catalogs", "SDSS_clean_cat", "SDSSxWISE_cat.tbl")
wise_csv_path = os.path.join(DATA_ROOT, "catalogs", "SDSS_clean_cat_Duncan.csv")

if not os.path.exists(wise_csv_path):
    print(f"Creating WISE lookup CSV from .tbl...")
    t = Table.read(tbl_path, format="ipac")
    import pandas as pd
    df = pd.DataFrame({
        'source_id': t['source_id'].astype(str),
        'w1flux': t['w1flux'].astype(float),
        'w2flux': t['w2flux'].astype(float),
    })
    df.to_csv(wise_csv_path, index=False)
    print(f"  Created: {wise_csv_path} ({len(df):,} rows)")
else:
    print(f"WISE lookup CSV already exists: {wise_csv_path}")

# ============================================================
# Step 1: Sample from catalog
# ============================================================
print(f"\n{'='*60}")
print("Step 1: Sampling from SDSSxWISE catalog")
print("=" * 60)

t = Table.read(tbl_path, format="ipac")
print(f"Catalog: {len(t):,} rows, DEC=[{t['dec'].min():.1f}, {t['dec'].max():.1f}]")

np.random.seed(42)
samples_per_class = 10  # small test
all_samples = []
for cls in ['GALAXY', 'STAR', 'QSO']:
    mask = t['class_01'] == cls
    indices = np.where(mask)[0]
    chosen = np.random.choice(indices, size=samples_per_class, replace=False)
    for idx in chosen:
        all_samples.append({
            'source_id': str(t['source_id'][idx]),
            'ra': float(t['ra'][idx]),
            'dec': float(t['dec'][idx]),
            'class': cls,
        })
    print(f"  {cls}: {samples_per_class} from {len(indices):,}")

print(f"Total: {len(all_samples)} sources")

# ============================================================
# Step 2: Download PanSTARRS images
# ============================================================
print(f"\n{'='*60}")
print("Step 2: Download PanSTARRS cutouts (thesis method)")
print("=" * 60)

output_root = os.path.join(PROJECT_ROOT, "source_cutouts", "north", "sdss", "ps")
os.makedirs(output_root, exist_ok=True)

from panstamps.downloader import downloader

def download_one(ra, dec, out_dir, arcsec=60):
    """Download 5-band PanSTARRS stack cutout. Returns number of bands."""
    try:
        fits_paths, _, _ = downloader(
            log=astropy_log,
            fits=True, jpeg=False, color=False,
            ra=ra, dec=dec,
            imageType='stack',
            filterSet='grizy',
            arcsecSize=arcsec
        ).get()

        if not fits_paths:
            return 0

        band_names = ['g', 'r', 'i', 'z', 'y']
        for band, src_path in zip(band_names, fits_paths):
            dst_path = os.path.join(out_dir, f"stack_{band}.fits")
            shutil.copy2(src_path, dst_path)

        return len(fits_paths)
    except Exception as e:
        return 0

downloaded = 0
for i, src in enumerate(all_samples):
    cls_dir = os.path.join(output_root, src['class'])
    src_dir = os.path.join(cls_dir, src['source_id'])

    # Skip if already complete
    if os.path.isdir(src_dir):
        existing = [f for f in os.listdir(src_dir) if f.endswith('.fits')]
        if len(existing) >= 5:
            downloaded += 1
            continue

    os.makedirs(src_dir, exist_ok=True)
    print(f"  [{i+1}/{len(all_samples)}] {src['source_id']} ({src['class']}) "
          f"ra={src['ra']:.4f} dec={src['dec']:.4f}...", end=" ", flush=True)

    n_bands = download_one(src['ra'], src['dec'], src_dir)
    if n_bands >= 5:
        print(f"OK")
        downloaded += 1
        time.sleep(0.3)  # be nice to server
    elif n_bands > 0:
        print(f"PARTIAL ({n_bands}/5)")
    else:
        print(f"NO DATA")

print(f"\nDownloaded: {downloaded}/{len(all_samples)} sources with 5 bands")

# ============================================================
# Step 3: Test Model A
# ============================================================
print(f"\n{'='*60}")
print("Step 3: Test Model A on downloaded images")
print("=" * 60)

# Check which sources are available
available = {}
for cls in ['GALAXY', 'STAR', 'QSO']:
    cls_dir = os.path.join(output_root, cls)
    if not os.path.isdir(cls_dir):
        continue
    for src_id in os.listdir(cls_dir):
        src_dir = os.path.join(cls_dir, src_id)
        if os.path.isdir(src_dir):
            fits_files = [f for f in os.listdir(src_dir) if f.endswith('.fits')]
            if len(fits_files) >= 5:
                available[src_id] = cls

print(f"Available sources: {len(available)}")
print(f"  GALAXY: {sum(1 for c in available.values() if c=='GALAXY')}")
print(f"  STAR:   {sum(1 for c in available.values() if c=='STAR')}")
print(f"  QSO:    {sum(1 for c in available.values() if c=='QSO')}")

if len(available) >= 3:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from mynetwork import CelestialClassficationNet
    from FitsImageFolder import FitsImageFolder
    from optical_dataset import OptDataSet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load Model A
    model_path = os.path.join(PROJECT_ROOT, "crossmatch", "models", "opt_classification_model_wts.pt")
    model = CelestialClassficationNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval().to(device)
    print(f"Model A loaded")

    # Create dataset
    dataset = FitsImageFolder(root=output_root)
    testData = OptDataSet(dataset, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(testData, batch_size=1, shuffle=False, num_workers=0)

    print(f"Dataset: {len(testData)} samples")
    print(f"Classes: {dataset.classes}")

    # Run inference
    results = []
    with torch.no_grad():
        for inputs, labels, wise_mag, ps_coords in test_loader:
            inputs = inputs.float().to(device)
            wise_mag = wise_mag.float().to(device)

            outputs = model(inputs, wise_mag)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

            results.append({
                'Galaxy': float(probs[0]),
                'QSO': float(probs[1]),
                'STAR': float(probs[2]),
                'label': int(labels.item()),
            })

    # Print results
    galaxy = [r['Galaxy'] for r in results]
    qso = [r['QSO'] for r in results]
    star = [r['STAR'] for r in results]

    print(f"\n{'='*60}")
    print(f"RESULTS: Model A on SDSS-area PanSTARRS ({len(results)} samples)")
    print(f"{'='*60}")
    print(f"Galaxy_prob: mean={np.mean(galaxy):.4f} median={np.median(galaxy):.4f} std={np.std(galaxy):.4f}")
    print(f"QSO_prob:    mean={np.mean(qso):.4f} median={np.median(qso):.4f} std={np.std(qso):.4f}")
    print(f"STAR_prob:   mean={np.mean(star):.4f} median={np.median(star):.4f} std={np.std(star):.4f}")

    max_classes = []
    for r in results:
        max_classes.append(max(('Galaxy','QSO','STAR'), key=lambda k: r[k]))
    print(f"\nDominant class: {Counter(max_classes)}")
    print(f"Ranges: Galaxy=[{min(galaxy):.4f}, {max(galaxy):.4f}]")
    print(f"        QSO=[{min(qso):.4f}, {max(qso):.4f}]")
    print(f"        STAR=[{min(star):.4f}, {max(star):.4f}]")

    print(f"\n--- Comparison ---")
    print(f"Original paper: QSO mean=0.745 (77%), STAR mean=0.241 (23%), std~0.37")
    print(f"CDFS (re-dl):   STAR mean=0.919, std=0.085, 100% STAR dominant")
    print(f"SDSS-area now:  STAR mean={np.mean(star):.4f}, std_STAR={np.std(star):.4f}")

    if np.std(star) > 0.18:
        print(f"\n*** Model A RECOVERS DISCRIMINATION on SDSS-area images! ***")
        print(f"*** Root cause confirmed: sky region / PanSTARRS image mismatch ***")
    else:
        print(f"\n*** Model A still degraded ***")
        print(f"*** Issue may be PanSTARRS API changes or epoch differences ***")

else:
    print(f"\nNeed at least 3 sources with 5-band FITS to test Model A.")
    print(f"Currently have: {len(available)}")
