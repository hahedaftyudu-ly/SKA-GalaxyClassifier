"""
North-sky Model B validation.
Uses FitsImageFolder for correct PS+WISE preprocessing (remove_nan, optimize_image, cal_luptitude).
VLASS via cadc.get_images(), PanSTARRS via panstamps.downloader.
"""
import csv, os, sys, time, warnings, shutil, glob
import numpy as np

warnings.filterwarnings("ignore")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "crossmatch"))
# tools/ must be last to avoid shadowing project utils.py
sys.path.append(os.path.join(PROJ, "tools"))

DATA = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJ, "data"))
RADIO_DIR = os.path.join(PROJ, "source_cutouts", "north", "rgz", "radio")
PS_DIR = os.path.join(PROJ, "source_cutouts", "north", "rgz", "ps")
RESULTS_DIR = os.path.join(PROJ, "crossmatch", "training_notes")
os.makedirs(RADIO_DIR, exist_ok=True)
os.makedirs(PS_DIR, exist_ok=True)


def sample_rgz_sources(n=10, random_state=42):
    host_path = os.path.join(DATA, "catalogs", "DR1_FIRST_host_properties.csv")
    radio_path = os.path.join(DATA, "catalogs", "DR1_FIRST_radio_classifications.csv")
    radio_by_cat = {}
    with open(radio_path) as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["CatID"])
                if int(row["N_votes"]) >= 10 and float(row["CL"]) >= 0.7:
                    radio_by_cat[cid] = {"ra": float(row["RA"]), "dec": float(row["Dec"])}
            except (ValueError, KeyError): continue
    sources = []
    with open(host_path) as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["#CatID"])
                hra = float(row["Host_RA"]); hdec = float(row["Host_Dec"])
                w1 = float(row["W1"]); w2 = float(row["W2"])
                if hra == 0 or cid not in radio_by_cat: continue
                r = radio_by_cat[cid]
                if r["dec"] < 10.0: continue
                sources.append({
                    "cat_id": cid, "rgz_id": row["RGZID"],
                    "radio_ra": r["ra"], "radio_dec": r["dec"],
                    "host_ra": hra, "host_dec": hdec,
                    "w1_mag": w1, "w2_mag": w2,
                })
            except (ValueError, KeyError): continue
    rng = np.random.RandomState(random_state)
    sampled = rng.choice(sources, size=min(n, len(sources)), replace=False)
    print(f"[Step 1] Sampled {len(sampled)}/{len(sources)} RGZ FIRST sources")
    return [dict(s) for s in sampled]


def download_vlass(sources):
    from astroquery.cadc import Cadc
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cadc = Cadc()
    results = {}
    for i, src in enumerate(sources):
        outpath = os.path.join(RADIO_DIR, f"vlass_{src['cat_id']}.fits")
        if os.path.exists(outpath):
            results[src["cat_id"]] = outpath; continue
        coord = SkyCoord(src["radio_ra"], src["radio_dec"], unit="deg")
        try:
            images = cadc.get_images(coord, radius=0.02 * u.deg, collection="VLASS")
        except Exception as e:
            print(f"  [{i+1}] VLASS error: {type(e).__name__}"); continue
        if not images: continue
        images[0].writeto(outpath, overwrite=True)
        results[src["cat_id"]] = outpath
        print(f"  [{i+1}/{len(sources)}] VLASS: {src['rgz_id']}")
        time.sleep(0.2)
    print(f"  VLASS: {len(results)}/{len(sources)}")
    return results


def download_ps(sources):
    from panstamps.downloader import downloader
    from astropy.logger import log as alog
    results = {}
    for i, src in enumerate(sources):
        src_id = f"rgz_{src['cat_id']}"
        src_dir = os.path.join(PS_DIR, src_id)
        os.makedirs(src_dir, exist_ok=True)
        fits_count = len([f for f in os.listdir(src_dir) if f.endswith(".fits")])
        if fits_count >= 5:
            results[src["cat_id"]] = src_dir
            continue
        tmpdir = os.path.join(PS_DIR, "_tmp"); os.makedirs(tmpdir, exist_ok=True)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            dl = downloader(log=alog, fits=True, jpeg=False, color=False,
                            ra=src["host_ra"], dec=src["host_dec"],
                            imageType='stack', filterSet='grizy', arcsecSize=60)
            fits_paths, _, _ = dl.get()
        finally:
            os.chdir(old_cwd)
        for fp in fits_paths:
            sp = os.path.join(tmpdir, fp)
            if os.path.exists(sp): shutil.move(sp, os.path.join(src_dir, os.path.basename(fp)))
        fits_count = len([f for f in os.listdir(src_dir) if f.endswith(".fits")])
        if fits_count >= 5:
            results[src["cat_id"]] = src_dir
            print(f"  [{i+1}/{len(sources)}] PS: {src_id}")
        else:
            print(f"  [{i+1}/{len(sources)}] PS partial: {fits_count}/5")
        time.sleep(0.5)
    print(f"  PS: {len(results)}/{len(sources)}")
    return results


def run_inference(sources, radio_results, ps_results):
    import torch
    import torch.nn.functional as F
    from mynetwork import CelestialClassficationNet
    from crossmatch.model_crossmatch import RadioOpticalCrossmatchModel
    from FitsImageFolder import FitsImageFolder
    from utils import cal_luptitude
    from astropy.io import fits
    from torchvision import transforms
    import PIL.Image as Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ---- Create FitsImageFolder directory ----
    tmp_dir = os.path.join(PROJ, "source_cutouts", ".tmp_north")
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
    os.makedirs(os.path.join(tmp_dir, "UNKNOWN"), exist_ok=True)
    cat_to_src = {}
    for src in sources:
        if src["cat_id"] not in ps_results: continue
        cat_to_src[src["cat_id"]] = src
        dst = os.path.join(tmp_dir, "UNKNOWN", f"rgz_{src['cat_id']}")
        if not os.path.exists(dst):
            try: os.symlink(os.path.abspath(ps_results[src["cat_id"]]), dst)
            except OSError: shutil.copytree(ps_results[src["cat_id"]], dst)

    # ---- Model A via FitsImageFolder (correct preprocessing) ----
    print("  Loading Model A...")
    model_a = CelestialClassficationNet()
    model_a.load_state_dict(torch.load(
        os.path.join(PROJ, "crossmatch", "models", "opt_classification_model_wts.pt"),
        map_location=device, weights_only=True))
    model_a.to(device).eval()

    dataset_a = FitsImageFolder(root=tmp_dir)
    print(f"  Model A: {len(dataset_a)} sources loaded via FitsImageFolder")

    def mag2flux(mag, zp=309.540):
        return zp * 10**(-mag / 2.5) * 1000

    model_a_results = {}
    for i in range(len(dataset_a)):
        sample_path = dataset_a.samples[i][0]
        cat_id = int(os.path.basename(sample_path).replace("rgz_", ""))
        src = cat_to_src.get(cat_id)
        if src is None: continue

        # Use FitsImageFolder loader: remove_nan + optimize_image
        img_dat, _, _ = dataset_a.loader(sample_path)
        img_t = torch.from_numpy(img_dat).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # WISE via cal_luptitude (asinh magnitudes as the model expects)
        w1f, w2f = mag2flux(src["w1_mag"]), mag2flux(src["w2_mag"])
        wise_vals = cal_luptitude(w1f, w2f)
        wise_t = torch.tensor(wise_vals, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model_a(img_t, wise_t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
        model_a_results[cat_id] = {"galaxy_prob": float(probs[0]), "qso_prob": float(probs[1]), "star_prob": float(probs[2])}

    ga = [r["galaxy_prob"] for r in model_a_results.values()]
    qa = [r["qso_prob"] for r in model_a_results.values()]
    sa = [r["star_prob"] for r in model_a_results.values()]
    print(f"  Model A: Galaxy={np.mean(ga):.4f} QSO={np.mean(qa):.4f} STAR={np.mean(sa):.4f} std={np.std(sa):.4f}")

    # ---- Model B ----
    print("  Loading Model B...")
    model_b = RadioOpticalCrossmatchModel(pos_dims=3)
    model_b.load_state_dict(torch.load(
        os.path.join(PROJ, "crossmatch", "models", "RGZ_all_negative_crossmatch_model_wts.pt"),
        map_location=device, weights_only=True))
    model_b.to(device).eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(216)])

    final_results = []
    for src in sources:
        cat_id = src["cat_id"]
        if cat_id not in radio_results or cat_id not in model_a_results: continue

        # Radio
        try:
            with fits.open(radio_results[cat_id]) as hdul:
                rdata = hdul[0].data
            if rdata.ndim >= 4: rdata = rdata[0, 0]
            elif rdata.ndim == 3: rdata = rdata[0]
            rdata = np.nan_to_num(rdata.astype(np.float64), nan=0.0)
            rmin, rmax = rdata.min(), rdata.max()
            if rmax > rmin: rdata = (rdata - rmin) / (rmax - rmin)
            h, w = rdata.shape
            if h < 216 or w < 216:
                ph, pw = (216-h)//2, (216-w)//2
                rdata = np.pad(rdata, ((ph, 216-h-ph), (pw, 216-w-pw)), mode='constant')
            rimg = Image.fromarray((rdata * 255).astype(np.uint8)).convert('L')
            rt = transform(rimg)
            if rt.shape[0] == 3: rt = rt.mean(dim=0, keepdim=True)
            rt = rt.unsqueeze(0).to(device)
        except Exception: continue

        # PS via FitsImageFolder loader + resize
        try:
            sample_path = os.path.join(tmp_dir, "UNKNOWN", f"rgz_{cat_id}")
            img_dat, _, _ = dataset_a.loader(sample_path)
            ps_t = torch.from_numpy(img_dat).permute(2, 0, 1).float().unsqueeze(0).to(device)
        except Exception: continue

        # Model A probs
        ma = model_a_results[cat_id]
        a_t = torch.tensor([[ma["galaxy_prob"], ma["qso_prob"], ma["star_prob"]]], dtype=torch.float32, device=device)

        # Position
        dra = (src["host_ra"] - src["radio_ra"]) * np.cos(np.radians((src["host_dec"]+src["radio_dec"])/2)) * 3600
        pos = torch.tensor([[dra]], dtype=torch.float32, device=device)
        zero = torch.tensor([[0.0]], dtype=torch.float32, device=device)

        with torch.no_grad():
            b_out = model_b(rt, ps_t, a_t, pos, zero, zero)
            b_probs = F.softmax(b_out, dim=1).cpu().numpy()[0]

        final_results.append({
            "rgz_id": src["rgz_id"], "cat_id": cat_id,
            "radio_ra": src["radio_ra"], "radio_dec": src["radio_dec"],
            "host_ra": src["host_ra"], "host_dec": src["host_dec"],
            "galaxy_prob": ma["galaxy_prob"], "qso_prob": ma["qso_prob"], "star_prob": ma["star_prob"],
            "nonhost_prob": float(b_probs[0]), "host_prob": float(b_probs[1]),
            "ground_truth": 1, "prediction": int(b_probs[1] > 0.5),
        })

    print(f"  Model B: {len(final_results)} sources processed")
    return final_results


def evaluate(results):
    if not results:
        print("  No results!"); return
    tp = sum(1 for r in results if r["prediction"] == 1)
    fn = sum(1 for r in results if r["prediction"] == 0)
    n = len(results)
    host_probs = [r["host_prob"] for r in results]
    ga = [r["galaxy_prob"] for r in results]
    qa = [r["qso_prob"] for r in results]
    sa = [r["star_prob"] for r in results]
    print(f"\n{'='*60}")
    print(f"NORTH SKY Model B Validation - {n} sources (all RGZ-confirmed hosts)")
    print(f"{'='*60}")
    print(f"  Host Recall:   {tp/(tp+fn)*100:.1f}% ({tp}/{n})" if (tp+fn)>0 else "  N/A")
    print(f"  Host_prob:     mean={np.mean(host_probs):.4f} max={np.max(host_probs):.4f} min={np.min(host_probs):.4f}")
    print(f"  Model A:       Galaxy={np.mean(ga):.4f} QSO={np.mean(qa):.4f} STAR={np.mean(sa):.4f} std={np.std(sa):.4f}")
    print(f"\n  {'RGZ ID':<30} {'Host':>8} {'Gal':>6} {'QSO':>6} {'STAR':>6}")
    print(f"  {'-'*60}")
    for r in sorted(results, key=lambda x: x["host_prob"], reverse=True):
        print(f"  {r['rgz_id']:<30} {r['host_prob']:>7.4f} {r['galaxy_prob']:>5.3f} {r['qso_prob']:>5.3f} {r['star_prob']:>5.3f}")
    csv_path = os.path.join(RESULTS_DIR, "north_sky_validation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print(f"\n  Saved: {csv_path}")


def main(n=10, skip_download=False):
    print("=" * 60)
    print("NORTH SKY Model B Validation")
    print("=" * 60)
    sources = sample_rgz_sources(n=n)
    if not skip_download:
        print("\n[Step 2] VLASS cutouts...")
        radio = download_vlass(sources)
        print("\n[Step 3] PanSTARRS cutouts...")
        ps = download_ps(sources)
    else:
        radio = {}; ps = {}
        for f in os.listdir(RADIO_DIR):
            if f.endswith(".fits"):
                for s in sources:
                    if str(s["cat_id"]) in f: radio[s["cat_id"]] = os.path.join(RADIO_DIR, f)
        for d in os.listdir(PS_DIR):
            try:
                cid = int(d.replace("rgz_", ""))
                dp = os.path.join(PS_DIR, d)
                if len([f for f in os.listdir(dp) if f.endswith(".fits")]) >= 5: ps[cid] = dp
            except ValueError: pass
    valid = sum(1 for s in sources if s["cat_id"] in radio and s["cat_id"] in ps)
    print(f"\n  Valid (VLASS+PS): {valid}/{len(sources)}")
    if valid < 2:
        print("  Need >=2. Run without --skip_download first."); return
    print("\n[Step 4] Model A -> B inference...")
    results = run_inference(sources, radio, ps)
    print("\n[Step 5] Evaluation...")
    evaluate(results)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--skip_download", action="store_true")
    args = p.parse_args()
    main(n=args.n, skip_download=args.skip_download)
