"""
XGBoost 模型A -> 模型B 端到端评估 (北天 RGZ 源, 完全离线)

与 north_sky_validate.py 相同的采样和推理流程, 唯一区别:
模型A的三分类概率来自 XGBoost (WISE-only 特征), 而非 CNN 切图模型。

对比口径: 7 个 RGZ 确认宿主, CNN-A 版 Recall=100%, host_prob 0.608-0.615
"""
import csv
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "crossmatch"))
sys.path.append(os.path.join(PROJ, "tools"))

DATA = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJ, "data"))
RADIO_DIR = os.path.join(PROJ, "source_cutouts", "north", "rgz", "radio")
PS_DIR = os.path.join(PROJ, "source_cutouts", "north", "rgz", "ps")
MODEL_PATH = os.path.join(DATA, "cache", "xgb_model_a.json")

# 与 north_sky_validate.sample_rgz_sources 相同的采样逻辑
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
            except (ValueError, KeyError):
                continue
    sources = []
    with open(host_path) as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["#CatID"])
                hra = float(row["Host_RA"]); hdec = float(row["Host_Dec"])
                w1 = float(row["W1"]); w2 = float(row["W2"])
                if hra == 0 or cid not in radio_by_cat:
                    continue
                r = radio_by_cat[cid]
                if r["dec"] < 10.0:
                    continue
                sources.append({
                    "cat_id": cid, "rgz_id": row["RGZID"],
                    "radio_ra": r["ra"], "radio_dec": r["dec"],
                    "host_ra": hra, "host_dec": hdec,
                    "w1_mag": w1, "w2_mag": w2,
                })
            except (ValueError, KeyError):
                continue
    rng = np.random.RandomState(random_state)
    sampled = rng.choice(sources, size=min(n, len(sources)), replace=False)
    print(f"[sample] {len(sampled)}/{len(sources)} RGZ FIRST sources")
    return [dict(s) for s in sampled]


def xgb_predict(model, src):
    """WISE 星等特征 -> [GALAXY, QSO, STAR] 概率 (与训练特征一致)"""
    w1, w2 = src["w1_mag"], src["w2_mag"]
    feat = np.array([[w1, w2, w1 - w2]])
    return model.predict_proba(feat)[0]


def main(n=10):
    print("=" * 60)
    print("XGBOOST-A -> Model B END-TO-END (north RGZ)")
    print("=" * 60)

    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"[model] loaded {MODEL_PATH}")

    sources = sample_rgz_sources(n=n)

    # 本地切图
    radio = {}
    for f in os.listdir(RADIO_DIR):
        if f.endswith(".fits"):
            for s in sources:
                if str(s["cat_id"]) in f:
                    radio[s["cat_id"]] = os.path.join(RADIO_DIR, f)
    ps = {}
    for d in os.listdir(PS_DIR):
        try:
            cid = int(d.replace("rgz_", ""))
            dp = os.path.join(PS_DIR, d)
            if len([f for f in os.listdir(dp) if f.endswith(".fits")]) >= 5:
                ps[cid] = dp
        except ValueError:
            pass
    valid = [s for s in sources if s["cat_id"] in radio and s["cat_id"] in ps]
    print(f"[cutouts] valid (VLASS+PS): {len(valid)}/{len(sources)}")
    if len(valid) < 2:
        print("  Not enough local cutouts. Abort.")
        return

    import torch
    import torch.nn.functional as F
    from crossmatch.model_crossmatch import RadioOpticalCrossmatchModel
    from FitsImageFolder import FitsImageFolder
    from astropy.io import fits
    from torchvision import transforms
    import PIL.Image as Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    model_b = RadioOpticalCrossmatchModel(pos_dims=3)
    model_b.load_state_dict(torch.load(
        os.path.join(PROJ, "crossmatch", "models", "RGZ_all_negative_crossmatch_model_wts.pt"),
        map_location=device, weights_only=True))
    model_b.to(device).eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(216)])

    # FitsImageFolder 装载 PS 切图 (与 CNN 版相同的预处理)
    tmp_dir = os.path.join(PROJ, "source_cutouts", ".tmp_xgb_eval")
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)
    os.makedirs(os.path.join(tmp_dir, "UNKNOWN"), exist_ok=True)
    import shutil as sh
    for s in valid:
        dst = os.path.join(tmp_dir, "UNKNOWN", f"rgz_{s['cat_id']}")
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(ps[s["cat_id"]]), dst)
            except OSError:
                sh.copytree(ps[s["cat_id"]], dst)
    dataset = FitsImageFolder(root=tmp_dir)

    results = []
    for s in valid:
        cid = s["cat_id"]
        # ---- Model A: XGBoost ----
        a_probs = xgb_predict(model, s)
        a_t = torch.tensor([a_probs], dtype=torch.float32, device=device)

        # ---- Radio cutout ----
        try:
            with fits.open(radio[cid]) as hdul:
                rdata = hdul[0].data
            if rdata.ndim >= 4:
                rdata = rdata[0, 0]
            elif rdata.ndim == 3:
                rdata = rdata[0]
            rdata = np.nan_to_num(rdata.astype(np.float64), nan=0.0)
            rmin, rmax = rdata.min(), rdata.max()
            if rmax > rmin:
                rdata = (rdata - rmin) / (rmax - rmin)
            h, w = rdata.shape
            if h < 216 or w < 216:
                ph, pw = (216 - h) // 2, (216 - w) // 2
                rdata = np.pad(rdata, ((ph, 216 - h - ph), (pw, 216 - w - pw)), mode='constant')
            rimg = Image.fromarray((rdata * 255).astype(np.uint8)).convert('L')
            rt = transform(rimg)
            if rt.shape[0] == 3:
                rt = rt.mean(dim=0, keepdim=True)
            rt = rt.unsqueeze(0).to(device)
        except Exception:
            continue

        # ---- PS cutout ----
        try:
            sample_path = os.path.join(tmp_dir, "UNKNOWN", f"rgz_{cid}")
            img_dat, _, _ = dataset.loader(sample_path)
            ps_t = torch.from_numpy(img_dat).permute(2, 0, 1).float().unsqueeze(0).to(device)
        except Exception:
            continue

        # ---- Position ----
        dra = (s["host_ra"] - s["radio_ra"]) * np.cos(np.radians((s["host_dec"] + s["radio_dec"]) / 2)) * 3600
        pos = torch.tensor([[dra]], dtype=torch.float32, device=device)
        zero = torch.tensor([[0.0]], dtype=torch.float32, device=device)

        with torch.no_grad():
            b_out = model_b(rt, ps_t, a_t, pos, zero, zero)
            b_probs = F.softmax(b_out, dim=1).cpu().numpy()[0]

        results.append({
            "rgz_id": s["rgz_id"], "cat_id": cid,
            "galaxy_prob": float(a_probs[0]), "qso_prob": float(a_probs[1]), "star_prob": float(a_probs[2]),
            "host_prob": float(b_probs[1]), "ground_truth": 1,
            "prediction": int(b_probs[1] > 0.5),
        })

    # ---- 评估 ----
    tp = sum(1 for r in results if r["prediction"] == 1)
    fn = sum(1 for r in results if r["prediction"] == 0)
    print(f"\n{'='*60}")
    print(f"XGBOOST-A -> B: {len(results)} sources (all RGZ-confirmed hosts)")
    print(f"{'='*60}")
    print(f"  Host Recall:   {tp/(tp+fn)*100:.1f}% ({tp}/{tp+fn})")
    print(f"  Host_prob:     mean={np.mean([r['host_prob'] for r in results]):.4f} "
          f"max={np.max([r['host_prob'] for r in results]):.4f} "
          f"min={np.min([r['host_prob'] for r in results]):.4f}")
    print(f"  Model A(XGB):  Galaxy={np.mean([r['galaxy_prob'] for r in results]):.4f} "
          f"QSO={np.mean([r['qso_prob'] for r in results]):.4f} "
          f"STAR={np.mean([r['star_prob'] for r in results]):.4f}")
    print(f"\n  {'RGZ ID':<30} {'Host':>8} {'Gal':>6} {'QSO':>6} {'STAR':>6}")
    print(f"  {'-'*60}")
    for r in sorted(results, key=lambda x: x["host_prob"], reverse=True):
        print(f"  {r['rgz_id']:<30} {r['host_prob']:>7.4f} "
              f"{r['galaxy_prob']:>5.3f} {r['qso_prob']:>5.3f} {r['star_prob']:>5.3f}")

    out_path = os.path.join(PROJ, "crossmatch", "training_notes", "xgb_model_a_north_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main(n=10)
