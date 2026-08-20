"""
crossmatch_rgz_labels.py — 将RGZ DR1 FIRST标签交叉匹配到我们的数据

RGZ DR1 FIRST数据 (北天, 与我们模型训练区域一致):
  - DR1_FIRST_radio_classifications.csv: 射电形态分类 (公民科学投票)
  - DR1_FIRST_host_properties.csv: 宿主星系属性 + WISE星等

用途:
  1. 匹配到Session5北天测试源 (source_cutouts/north/sdss/ps/)
  2. 匹配到preprocessed_cat中的训练/测试CSV
  3. 给模型B提供独立的公民科学GT标签

用法:
  python tools/crossmatch_rgz_labels.py
"""

import csv
import os
import sys
from math import cos, radians

import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJ_ROOT, "data"))


def load_rgz_radio():
    """加载RGZ FIRST射电分类表"""
    path = os.path.join(DATA_DIR, "catalogs", "DR1_FIRST_radio_classifications.csv")
    sources = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sources.append({
                    "cat_id": int(row["CatID"]),
                    "rgz_id": row["RGZID"],
                    "zooniverse_id": row["ZooniverseID"],
                    "ra": float(row["RA"]),
                    "dec": float(row["Dec"]),
                    "n_votes": int(row["N_votes"]),
                    "n_total": int(row["N_total"]),
                    "consensus": float(row["CL"]),
                    "n_comp": int(row["N_comp"]),
                    "n_peaks": int(row["N_peaks"]),
                })
            except (ValueError, KeyError):
                continue
    print(f"Loaded {len(sources)} RGZ FIRST radio sources")
    return sources


def load_rgz_host():
    """加载RGZ FIRST宿主属性表 (含WISE星等和宿主位置)"""
    path = os.path.join(DATA_DIR, "catalogs", "DR1_FIRST_host_properties.csv")
    hosts = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                host_ra = float(row["Host_RA"]) if row["Host_RA"] else None
                host_dec = float(row["Host_Dec"]) if row["Host_Dec"] else None
                hosts.append({
                    "cat_id": int(row["#CatID"]),
                    "rgz_id": row["RGZID"],
                    "host_ra": host_ra,
                    "host_dec": host_dec,
                    "w1_mag": float(row["W1"]) if row["W1"] else None,
                    "w2_mag": float(row["W2"]) if row["W2"] else None,
                    "wise_id": row["WISEID"],
                })
            except (ValueError, KeyError):
                continue
    print(f"Loaded {len(hosts)} RGZ FIRST host entries")
    return hosts


def angdist(ra1, dec1, ra2, dec2):
    """角距离 (arcsec)"""
    dra = (ra1 - ra2) * cos(radians((dec1 + dec2) / 2))
    ddec = dec1 - dec2
    return np.sqrt(dra**2 + ddec**2) * 3600


def crossmatch_host_to_radio(hosts, radio_sources, max_sep=3.0):
    """匹配宿主到射电源 (通过CatID)"""
    # 主要按CatID匹配
    radio_by_cat = {r["cat_id"]: r for r in radio_sources}
    matched = 0
    results = []
    for h in hosts:
        r = radio_by_cat.get(h["cat_id"])
        if r and h["host_ra"] is not None:
            # 验证位置一致性
            sep = angdist(h["host_ra"], h["host_dec"], r["ra"], r["dec"])
            results.append({**h, **r, "radio_host_sep": sep})
            matched += 1
    print(f"Matched {matched} hosts to radio sources (of {len(hosts)} hosts, {len(radio_sources)} radio)")
    return results


def crossmatch_to_test_data(rgz_matched, csv_path, max_sep=5.0):
    """将RGZ匹配结果交叉匹配到测试CSV"""
    import csv as csv_module

    # Load test data
    test_sources = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                try:
                    test_sources.append({
                        "ra": float(row.get("ra_PS", row.get("RA", row.get("ra", 0)))),
                        "dec": float(row.get("dec_PS", row.get("DEC", row.get("dec", 0)))),
                        **row,
                    })
                except (ValueError, KeyError):
                    continue

    matches = []
    for ts in test_sources:
        best_sep = 999
        best_match = None
        for rm in rgz_matched:
            sep = angdist(ts["ra"], ts["dec"], rm["ra"], rm["dec"])
            if sep < best_sep:
                best_sep = sep
                best_match = rm
        if best_sep <= max_sep and best_match:
            matches.append({**ts, "rgz_match": best_match, "rgz_sep": best_sep})

    print(f"Crossmatched {len(matches)}/{len(test_sources)} test sources to RGZ")
    return matches


def main():
    print("=" * 60)
    print("RGZ DR1 FIRST → GalaxyClassifier Crossmatch")
    print("=" * 60)

    # 加载RGZ数据
    radio = load_rgz_radio()
    hosts = load_rgz_host()

    # 匹配宿主到射电源
    matched = crossmatch_host_to_radio(hosts, radio)
    if not matched:
        print("WARNING: No host-radio matches found. Check CatID alignment.")
        return

    # 统计
    has_host = [m for m in matched if m["host_ra"] is not None]
    has_wise = [m for m in matched if m["w1_mag"] is not None]
    print(f"  With host position: {len(has_host)}")
    print(f"  With WISE mag: {len(has_wise)}")
    print(f"  Median host-radio separation: {np.median([m['radio_host_sep'] for m in matched if m.get('radio_host_sep')]):.2f} arcsec")

    # 保存交叉匹配结果
    out_path = os.path.join(DATA_DIR, "rgz_first_matched.csv")
    with open(out_path, "w", newline="") as f:
        if matched:
            fieldnames = list(matched[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matched)
    print(f"Saved {len(matched)} matched entries → {out_path}")

    # 尝试匹配到preprocessed_cat中的CSV
    preproc_dir = os.path.join(DATA_DIR, "preprocessed_cat")
    if os.path.isdir(preproc_dir):
        print("\n--- Crossmatching to preprocessed catalogs ---")
        for csv_file in sorted(os.listdir(preproc_dir)):
            if csv_file.endswith(".csv"):
                csv_path = os.path.join(preproc_dir, csv_file)
                crossmatch_to_test_data(matched, csv_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
