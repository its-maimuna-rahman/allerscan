"""
Measure the photo scanner on YOUR photos.

Folder layout (default: scan_data/):
    nutella_back.jpg      the photo
    nutella_back.txt      what the label really says (type the ingredient list + any "may contain" line)
    nutella_back.allergens.json   (optional) e.g. ["hazelnut","cow_milk"]  - overrides the automatic answer

Run:
    python scan_benchmark.py                           # uses CONFIG defaults
    python scan_benchmark.py --engine both
    python scan_benchmark.py --engine easyocr --variant rgb
    python scan_benchmark.py --sweep-target-h 24,32,40 # tune the zoom size on your own photos
    python scan_benchmark.py --module photo_scanner_old --csv old.csv   # score your previous scanner the same way
Outputs a table and scan_report.csv.

The numbers that matter for an allergy app:
    missed allergens (FN)        a real allergen that is in neither the text nor the "uncertain" list  -> must be 0
    recall                       share of real allergens found in the extracted text
    recall incl. uncertain       share found by ANY reading (text + uncertain list)
    precision                    share of reported allergens that are real
"""
import argparse
import csv
import importlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_parser import parse_label

ps = None   # the scanner module under test, chosen by --module (default: photo_scanner)

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def words(text: str):
    return re.findall(r"[a-z]{3,}", text.lower())


def word_f1(truth: str, pred: str) -> float:
    t, p = words(truth), words(pred)
    if not t or not p:
        return 0.0
    from collections import Counter
    ct, cp = Counter(t), Counter(p)
    overlap = sum((ct & cp).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(t)
    return 2 * prec * rec / (prec + rec)


def expected_allergens(path: Path, truth: str) -> set:
    override = path.with_suffix(".allergens.json")
    if override.exists():
        return set(json.loads(override.read_text()))
    return set(parse_label(truth).keys())


def run_once(folder: Path, label: str):
    rows = []
    for img in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT):
        gt_file = img.with_suffix(".txt")
        if not gt_file.exists():
            continue
        truth = gt_file.read_text(encoding="utf-8")
        exp = expected_allergens(img, truth)
        t = time.time()
        try:
            res = ps.scan_photo(img.read_bytes(), mode="ingredient")
        except Exception as exc:  # noqa: BLE001
            res = {"success": False, "error": f"EXCEPTION {exc}"}
        secs = time.time() - t
        text = res.get("ingredients_text", "") if res.get("success") else ""
        got = set(parse_label(text).keys()) if text else set()
        unc = set(res.get("uncertain_allergens", []))
        tp, fp, fn = len(exp & got), len(got - exp), len(exp - got)
        fn_all = len(exp - (got | unc))
        rows.append({
            "run": label, "photo": img.name, "ok": int(bool(res.get("success"))),
            "expected": len(exp), "tp": tp, "fp": fp, "fn": fn, "fn_after_uncertain": fn_all,
            "missed": ",".join(sorted(exp - (got | unc))), "false_alarms": ",".join(sorted(got - exp)),
            "word_f1": round(word_f1(truth, text), 3), "rotation": res.get("rotation_applied", ""),
            "skew": res.get("skew_applied", ""), "seconds": round(secs, 2),
        })
    return rows


def summarize(rows, label):
    if not rows:
        return None
    tp, fp, fn = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows)
    fn_all = sum(r["fn_after_uncertain"] for r in rows)
    exp = sum(r["expected"] for r in rows)
    return {
        "run": label, "photos": len(rows), "read_ok": sum(r["ok"] for r in rows),
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "recall": round(tp / exp, 3) if exp else 0.0,
        "recall_incl_uncertain": round(1 - fn_all / exp, 3) if exp else 0.0,
        "missed_allergens": fn_all, "photos_with_a_miss": sum(1 for r in rows if r["fn_after_uncertain"] > 0),
        "mean_word_f1": round(statistics.mean(r["word_f1"] for r in rows), 3),
        "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="scan_data")
    ap.add_argument("--engine", choices=["easyocr", "tesseract", "both"])
    ap.add_argument("--variant", choices=["auto", "rgb", "clahe"])
    ap.add_argument("--sweep-target-h", help="comma list, e.g. 24,32,40")
    ap.add_argument("--module", default="photo_scanner", help="scanner module to test, e.g. photo_scanner_old")
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    default_csv = str(reports_dir / "scan_report.csv") if reports_dir.is_dir() else "scan_report.csv"
    ap.add_argument("--csv", default=default_csv)
    ap.add_argument("--verbose", action="store_true", help="print every photo with a miss or false alarm")
    a = ap.parse_args()
    global ps
    ps = importlib.import_module(a.module)
    cfg = getattr(ps, "CONFIG", None)
    if (a.engine or a.variant or a.sweep_target_h) and cfg is None:
        sys.exit(f"{a.module} has no CONFIG, so --engine/--variant/--sweep-target-h cannot be used with it.")
    if a.engine:
        cfg["engine"] = a.engine
    if a.variant:
        cfg["variant_mode"] = a.variant
    folder = Path(a.folder)
    if not folder.is_dir() and (Path(__file__).parent / a.folder).is_dir():
        folder = Path(__file__).parent / a.folder
    if not folder.is_dir() and (Path(__file__).parent.parent / a.folder).is_dir():
        folder = Path(__file__).parent.parent / a.folder
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    settings = [("default", None)]
    if a.sweep_target_h:
        settings = [(f"target_h={v}", int(v)) for v in a.sweep_target_h.split(",")]
    all_rows, summaries = [], []
    for label, th in settings:
        if th is not None:
            cfg["target_box_height"] = th
        rows = run_once(folder, label)
        all_rows += rows
        s = summarize(rows, label)
        if s:
            summaries.append(s)
        if a.verbose:
            for r in rows:
                if r["fn_after_uncertain"] or r["fp"] or not r["ok"]:
                    print(f"  [{label}] {r['photo']}: missed={r['missed'] or '-'} false_alarms={r['false_alarms'] or '-'} ok={r['ok']}")
    if not summaries:
        sys.exit("No photo+txt pairs found.")
    keys = list(summaries[0].keys())
    print("\n" + " | ".join(keys))
    for s in summaries:
        print(" | ".join(str(s[k]) for k in keys))
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-photo results written to {a.csv}")
    if any(s["missed_allergens"] for s in summaries):
        print("WARNING: at least one real allergen was missed. Fix those photos first (see 'missed' column).")


if __name__ == "__main__":
    main()
