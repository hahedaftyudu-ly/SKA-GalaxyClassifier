"""
SDSS source Model A->B validation (paper method path continuation).
Session 5 showed Model A recovers on SDSS sources. Now complete the A->B chain:
  1. Download VLASS at SDSS positions
  2. Crossmatch with RGZ DR1 for host labels
  3. Run Model A -> Model B
"""
import csv, os, sys, time, warnings, shutil
import numpy as np
warnings.filterwarnings("ignore")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "crossmatch"))
sys.path.append(os.path.join(PROJ, "tools"))

DATA = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJ, "data"))
PS_TEST = os.path.join(PROJ, "source_cutouts", "north", "sdss", "ps")
RADIO_SDSS = os.path.join(PROJ, "source_cutouts", "north", "sdss", "radio")
RESULTS_DIR = os.path.join(PROJ, "crossmatch", "training_notes")
os.makedirs(RADIO_SDSS, exist_ok=True)


def load_sdss_sources():
    """Load the 30 SDSS sources with PS cutouts from Session 5"""
    csv_path = os.path.join(DATA, "catalogs", "thesis_method_test_sample.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    sources = []
    seen = set()
    for cls in ["GALAXY", "QSO", "STAR"]:
        cls_dir = os.path.join(PS_TEST, cls)
        if not os.path.isdir(cls_dir): continue
        for sid in os.listdir(cls_dir):
            if not os.path.isdir(os.path.join(cls_dir, sid)): continue
            for r in rows:
                if r["source_id"] == sid and sid not in seen:
                    seen.add(sid)
                    sources.append({
                        "source_id": sid, "class": cls,
                        "ra": float(r["ra"]), "dec": float(r["dec"]),
                    })
                    break
    print(f"Loaded {len(sources)} SDSS sources with PS cutouts")
    return sources


def crossmatch_rgz(sources, max_sep_arcsec=3.0):
    """Crossmatch SDSS positions to RGZ DR1 FIRST hosts for labels"""
    rgz_path = os.path.join(DATA, "catalogs", "DR1_FIRST_host_properties.csv")
    rgz_radio_path = os.path.join(DATA, "catalogs", "DR1_FIRST_radio_classifications.csv")

    # Load RGZ radio positions
    radio_positions = {}
    with open(rgz_radio_path) as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["CatID"])
                radio_positions[cid] = {"ra": float(row["RA"]), "dec": float(row["Dec"])}
            except (ValueError, KeyError): continue

    # Load RGZ hosts and match
    def angdist(ra1, dec1, ra2, dec2):
        dra = (ra1 - ra2) * np.cos(np.radians((dec1 + dec2) / 2))
        ddec = dec1 - dec2
        return np.sqrt(dra**2 + ddec**2) * 3600

    host_positions = []
    with open(rgz_path) as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["#CatID"])
                hra = float(row["Host_RA"]); hdec = float(row["Host_Dec"])
                if hra == 0: continue
                r = radio_positions.get(cid)
                if r is None: continue
                host_positions.append({"cat_id": cid, "rgz_id": row["RGZID"],
                    "host_ra": hra, "host_dec": hdec,
                    "radio_ra": r["ra"], "radio_dec": r["dec"]})
            except (ValueError, KeyError): continue

    # Match SDSS sources to RGZ hosts
    matched = 0
    for s in sources:
        best_sep = 999
        best_rgz = None
        for h in host_positions:
            sep = angdist(s["ra"], s["dec"], h["host_ra"], h["host_dec"])
            if sep < best_sep:
                best_sep = sep
                best_rgz = h
        if best_sep <= max_sep_arcsec and best_rgz:
            s["rgz_host"] = best_rgz
            s["rgz_sep"] = best_sep
            s["host_label"] = 1  # Within RGZ host position -> IS a host
            matched += 1
        else:
            s["rgz_host"] = None
            s["host_label"] = 0  # No RGZ match -> NOT a host (or unlabeled)

    print(f"RGZ crossmatch: {matched}/{len(sources)} matched to RGZ hosts")
    return sources


def download_vlass(sources):
    """Download VLASS cutouts at SDSS positions"""
    from astroquery.cadc import Cadc
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    cadc = Cadc()
    results = {}
    for i, s in enumerate(sources):
        outpath = os.path.join(RADIO_SDSS, f"vlass_{s['source_id']}.fits")
        if os.path.exists(outpath):
            results[s["source_id"]] = outpath; continue

        coord = SkyCoord(s["ra"], s["dec"], unit="deg")
        try:
            images = cadc.get_images(coord, radius=0.02 * u.deg, collection="VLASS")
            if images:
                images[0].writeto(outpath, overwrite=True)
                results[s["source_id"]] = outpath
                print(f"  [{i+1}/{len(sources)}] VLASS OK: {s['source_id']} ({s['class']})")
            else:
                print(f"  [{i+1}/{len(sources)}] No VLASS: {s['source_id']}")
        except Exception as e:
            print(f"  [{i+1}/{len(sources)}] VLASS err: {type(e).__name__}")
        time.sleep(0.3)

    print(f"VLASS: {len(results)}/{len(sources)}")
    return results


def run_ab_inference(sources, radio_results):
    """Run Model A (FitsImageFolder) -> Model B on SDSS sources"""
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

    # Model A via FitsImageFolder
    print("  Loading Model A...")
    model_a = CelestialClassficationNet()
    model_a.load_state_dict(torch.load(
        os.path.join(PROJ, "crossmatch", "models", "opt_classification_model_wts.pt"),
        map_location=device, weights_only=True))
    model_a.to(device).eval()

    dataset_a = FitsImageFolder(root=PS_TEST)
    print(f"  Model A dataset: {len(dataset_a)} sources")

    model_a_results = {}
    for i in range(len(dataset_a)):
        sample_path = dataset_a.samples[i][0]
        source_id = os.path.basename(sample_path)
        img_dat, wise_mag, _ = dataset_a.loader(sample_path)
        img_t = torch.from_numpy(img_dat).permute(2, 0, 1).float().unsqueeze(0).to(device)
        wise_t = torch.from_numpy(wise_mag).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out = model_a(img_t, wise_t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
        model_a_results[source_id] = {"galaxy_prob": float(probs[0]), "qso_prob": float(probs[1]), "star_prob": float(probs[2])}

    ga = [r["galaxy_prob"] for r in model_a_results.values()]
    qa = [r["qso_prob"] for r in model_a_results.values()]
    sa = [r["star_prob"] for r in model_a_results.values()]
    print(f"  Model A: Gal={np.mean(ga):.4f} QSO={np.mean(qa):.4f} STAR={np.mean(sa):.4f} std={np.std(sa):.4f}")

    # Model B
    print("  Loading Model B...")
    model_b = RadioOpticalCrossmatchModel(pos_dims=3)
    model_b.load_state_dict(torch.load(
        os.path.join(PROJ, "crossmatch", "models", "RGZ_all_negative_crossmatch_model_wts.pt"),
        map_location=device, weights_only=True))
    model_b.to(device).eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(216)])

    final_results = []
    for s in sources:
        sid = s["source_id"]
        if sid not in radio_results or sid not in model_a_results: continue

        # Radio
        try:
            with fits.open(radio_results[sid]) as hdul:
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

        # Optical via FitsImageFolder loader
        try:
            sample_path = os.path.join(PS_TEST, s["class"], sid)
            img_dat, wise_mag, _ = dataset_a.loader(sample_path)
            ps_t = torch.from_numpy(img_dat).permute(2, 0, 1).float().unsqueeze(0).to(device)
        except Exception: continue

        # Model A probs
        ma = model_a_results[sid]
        a_t = torch.tensor([[ma["galaxy_prob"], ma["qso_prob"], ma["star_prob"]]], dtype=torch.float32, device=device)
        dra_t = torch.tensor([[0.0]], dtype=torch.float32, device=device)
        zero = torch.tensor([[0.0]], dtype=torch.float32, device=device)

        with torch.no_grad():
            b_out = model_b(rt, ps_t, a_t, dra_t, zero, zero)
            b_probs = F.softmax(b_out, dim=1).cpu().numpy()[0]

        final_results.append({
            "source_id": sid, "class": s["class"],
            "ra": s["ra"], "dec": s["dec"],
            "galaxy_prob": ma["galaxy_prob"], "qso_prob": ma["qso_prob"], "star_prob": ma["star_prob"],
            "nonhost_prob": float(b_probs[0]), "host_prob": float(b_probs[1]),
            "rgz_host_label": s.get("host_label", -1),
            "rgz_sep_arcsec": s.get("rgz_sep", -1),
            "prediction": int(b_probs[1] > 0.5),
        })

    print(f"  Model B: {len(final_results)} sources processed")
    return final_results


def evaluate(results):
    if not results: return
    n = len(results)
    host_probs = [r["host_prob"] for r in results]
    ga = [r["galaxy_prob"] for r in results]
    qa = [r["qso_prob"] for r in results]
    sa = [r["star_prob"] for r in results]

    # Those with RGZ host = GT=1
    with_label = [r for r in results if r["rgz_host_label"] >= 0]
    tp = sum(1 for r in with_label if r["rgz_host_label"] == 1 and r["prediction"] == 1)
    fn = sum(1 for r in with_label if r["rgz_host_label"] == 1 and r["prediction"] == 0)
    fp = sum(1 for r in with_label if r["rgz_host_label"] == 0 and r["prediction"] == 1)
    tn = sum(1 for r in with_label if r["rgz_host_label"] == 0 and r["prediction"] == 0)

    print(f"\n{'='*60}")
    print(f"SDSS A->B Validation ({n} sources)")
    print(f"{'='*60}")
    print(f"  Model A: Gal={np.mean(ga):.4f} QSO={np.mean(qa):.4f} STAR={np.mean(sa):.4f} std={np.std(sa):.4f}")
    print(f"  Model B host_prob: mean={np.mean(host_probs):.4f} max={np.max(host_probs):.4f} min={np.min(host_probs):.4f}")
    if with_label:
        n_label = len(with_label)
        acc = (tp+tn)/n_label*100 if n_label>0 else 0
        rec = tp/(tp+fn)*100 if(tp+fn)>0 else 0
        print(f"  With RGZ labels ({n_label}): Acc={acc:.1f}% Recall={rec:.1f}% TP={tp} FN={fn} FP={fp} TN={tn}")

    print(f"\n  {'Source':<30} {'Class':<8} {'Host':>8} {'Gal':>6} {'QSO':>6} {'STAR':>6} {'RGZ':>4}")
    print(f"  {'-'*70}")
    for r in sorted(results, key=lambda x: x["host_prob"], reverse=True)[:15]:
        print(f"  {r['source_id']:<30} {r['class']:<8} {r['host_prob']:>7.4f} {r['galaxy_prob']:>5.3f} {r['qso_prob']:>5.3f} {r['star_prob']:>5.3f} {r['rgz_host_label']:>4}")

    csv_path = os.path.join(RESULTS_DIR, "sdss_ab_validation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print(f"\n  Saved: {csv_path}")


def main():
    print("=" * 60)
    print("SDSS Source Model A->B Validation")
    print("=" * 60)

    sources = load_sdss_sources()

    print("\n[Step 1] Crossmatch SDSS -> RGZ DR1 for host labels...")
    sources = crossmatch_rgz(sources)

    print("\n[Step 2] Download VLASS cutouts...")
    radio = download_vlass(sources)

    valid = sum(1 for s in sources if s["source_id"] in radio)
    print(f"\n  Valid (VLASS ready): {valid}/{len(sources)}")

    if valid < 3:
        print("  Not enough VLASS data. CADC may be unstable.")
        return

    print("\n[Step 3] Model A -> B inference...")
    results = run_ab_inference(sources, radio)

    print("\n[Step 4] Evaluation...")
    evaluate(results)


if __name__ == "__main__":
    main()
