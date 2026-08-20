"""
South-sky Model A test: sample SDSS sources at different Dec ranges,
download PS cutouts, run Model A, show Dec-dependent degradation.
"""
import csv, os, sys, time, warnings, shutil
import numpy as np
warnings.filterwarnings("ignore")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "crossmatch"))
sys.path.append(os.path.join(PROJ, "tools"))

DATA = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJ, "data"))
PS_SOUTH = os.path.join(PROJ, "source_cutouts", "south_test", "regions")


def sample_sdss_by_dec(n_per_band=5):
    """Sample SDSS sources in 3 Dec bands: North, Mid, South"""
    import pandas as pd

    # Use thesis_method_test_sample.csv which has ra, dec, class_01
    csv_path = os.path.join(DATA, "catalogs", "thesis_method_test_sample.csv")
    df = pd.read_csv(csv_path)

    print(f"  Loaded {len(df)} entries from {csv_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dec range: {df['dec'].min():.1f} to {df['dec'].max():.1f}")
    print(f"  Classes: {df['class_01'].unique()}")

    bands = [
        ("North", 20, 85),
        ("Mid", 0, 20),
        ("South", -20, 0),
    ]

    samples = []
    for band_name, dec_min, dec_max in bands:
        band_df = df[(df['dec'] >= dec_min) & (df['dec'] < dec_max)]
        print(f"  {band_name} (Dec {dec_min} to {dec_max}): {len(band_df)} sources")

        for cls in ['GALAXY', 'QSO', 'STAR']:
            cls_df = band_df[band_df['class_01'] == cls]
            n_take = min(n_per_band, len(cls_df))
            if n_take == 0:
                print(f"    {cls}: 0 available, skipping")
                continue
            sampled = cls_df.sample(n=n_take, random_state=42) if len(cls_df) > n_take else cls_df
            for _, row in sampled.iterrows():
                samples.append({
                    'source_id': str(row['source_id']),
                    'class': cls,
                    'ra': float(row['ra']),
                    'dec': float(row['dec']),
                    'dec_band': band_name,
                })

    print(f"  Total sampled: {len(samples)} sources")
    return samples


def download_ps(sources):
    """Download PanSTARRS 5-band cutouts"""
    from panstamps.downloader import downloader
    from astropy.logger import log as alog

    os.makedirs(PS_SOUTH, exist_ok=True)
    results = {}

    for i, s in enumerate(sources):
        src_dir = os.path.join(PS_SOUTH, s['dec_band'], s['class'], s['source_id'])
        os.makedirs(src_dir, exist_ok=True)

        fits_count = len([f for f in os.listdir(src_dir) if f.endswith('.fits')]) if os.path.isdir(src_dir) else 0
        if fits_count >= 5:
            results[s['source_id']] = src_dir
            continue

        tmpdir = os.path.join(PS_SOUTH, "_tmp"); os.makedirs(tmpdir, exist_ok=True)
        old = os.getcwd()
        try:
            os.chdir(tmpdir)
            dl = downloader(log=alog, fits=True, jpeg=False, color=False,
                            ra=s['ra'], dec=s['dec'],
                            imageType='stack', filterSet='grizy', arcsecSize=60)
            paths, _, _ = dl.get()
        except Exception as e:
            os.chdir(old)
            print(f"  [{i+1}/{len(sources)}] {s['dec_band']}/{s['class']}: {type(e).__name__}")
            time.sleep(2)
            continue

        for fp in paths:
            sp = os.path.join(tmpdir, fp)
            if os.path.exists(sp):
                shutil.move(sp, os.path.join(src_dir, os.path.basename(fp)))

        fits_count = len([f for f in os.listdir(src_dir) if f.endswith('.fits')]) if os.path.isdir(src_dir) else 0
        if fits_count >= 5:
            results[s['source_id']] = src_dir
            print(f"  [{i+1}/{len(sources)}] {s['dec_band']}/{s['class']} OK")
        else:
            print(f"  [{i+1}/{len(sources)}] {s['dec_band']}/{s['class']}: {fits_count}/5 bands")
        time.sleep(0.5)

    print(f"  PS: {len(results)}/{len(sources)} ready")
    return results


def run_model_a(sources, ps_results):
    """Run Model A via FitsImageFolder"""
    import torch
    import torch.nn.functional as F
    from mynetwork import CelestialClassficationNet
    from FitsImageFolder import FitsImageFolder

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build FitsImageFolder structure
    tmp_dir = os.path.join(PROJ, "source_cutouts", ".tmp_south")
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)

    for s in sources:
        if s['source_id'] not in ps_results: continue
        cls_dir = os.path.join(tmp_dir, s['class'])
        os.makedirs(cls_dir, exist_ok=True)
        dst = os.path.join(cls_dir, s['source_id'])
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(ps_results[s['source_id']]), dst)
            except OSError:
                shutil.copytree(ps_results[s['source_id']], dst)

    model = CelestialClassficationNet()
    model.load_state_dict(torch.load(
        os.path.join(PROJ, "crossmatch", "models", "opt_classification_model_wts.pt"),
        map_location=device, weights_only=True))
    model.to(device).eval()

    dataset = FitsImageFolder(root=tmp_dir)
    print(f"  Dataset: {len(dataset)} sources, classes={dataset.classes}")

    # Build source_id -> sample index map
    sid_to_idx = {}
    for i in range(len(dataset)):
        sid = os.path.basename(dataset.samples[i][0])
        sid_to_idx[sid] = i

    results = []
    for s in sources:
        if s['source_id'] not in sid_to_idx: continue
        idx = sid_to_idx[s['source_id']]
        sample_path = dataset.samples[idx][0]
        img_dat, wise_mag, _ = dataset.loader(sample_path)
        img_t = torch.from_numpy(img_dat).permute(2, 0, 1).float().unsqueeze(0).to(device)
        wise_t = torch.from_numpy(wise_mag).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t, wise_t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]

        pred_cls = dataset.classes[int(probs.argmax())]
        results.append({
            'source_id': s['source_id'], 'class': s['class'],
            'dec_band': s['dec_band'], 'dec': s['dec'],
            'galaxy_prob': float(probs[0]), 'qso_prob': float(probs[1]), 'star_prob': float(probs[2]),
            'predicted': pred_cls, 'correct': int(pred_cls == s['class']),
        })

    return results


def evaluate(results):
    import numpy as np
    if not results: return

    print(f"\n{'='*65}")
    print(f"Model A: Dec-dependent Performance ({len(results)} sources)")
    print(f"{'='*65}")

    for band in ['North', 'Mid', 'South']:
        br = [r for r in results if r['dec_band'] == band]
        if not br: continue
        acc = sum(r['correct'] for r in br) / len(br) * 100
        ga = np.mean([r['galaxy_prob'] for r in br])
        qa = np.mean([r['qso_prob'] for r in br])
        sa = np.mean([r['star_prob'] for r in br])
        ss = np.std([r['star_prob'] for r in br])

        print(f"\n  {band} (Dec {min(r['dec'] for r in br):.0f} to {max(r['dec'] for r in br):.0f}):")
        print(f"    Accuracy: {acc:.0f}%")
        print(f"    Model A:  Gal={ga:.3f}  QSO={qa:.3f}  STAR={sa:.3f}  std={ss:.4f}")

        for cls in ['GALAXY', 'QSO', 'STAR']:
            cr = [r for r in br if r['class'] == cls]
            if cr:
                cacc = sum(r['correct'] for r in cr) / len(cr) * 100
                cga = np.mean([r['galaxy_prob'] for r in cr])
                print(f"    {cls}: acc={cacc:.0f}%  Gal={cga:.3f}")

    # Save
    out = os.path.join(PROJ, "crossmatch", "training_notes", "south_sky_model_a_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print(f"\n  Saved: {out}")


def main(n_per_band=5):
    print("=" * 65)
    print("South-Sky Model A Degradation Test")
    print("=" * 65)

    print("\n[Step 1] Sampling SDSS sources in 3 Dec bands...")
    sources = sample_sdss_by_dec(n_per_band=n_per_band)

    print("\n[Step 2] Downloading PanSTARRS cutouts...")
    ps = download_ps(sources)

    valid = sum(1 for s in sources if s['source_id'] in ps)
    print(f"\n  Valid: {valid}/{len(sources)}")

    if valid < 6:
        print("  Not enough data. Check panstamps.")
        return

    print("\n[Step 3] Running Model A...")
    results = run_model_a(sources, ps)

    print("\n[Step 4] Evaluation...")
    evaluate(results)


if __name__ == "__main__":
    main()
