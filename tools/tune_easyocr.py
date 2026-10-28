#!/usr/bin/env python3
"""
Tuning script for EasyOCR parameters on packaging photos.
Usage:
    python tune_easyocr.py <photo1> <truth1.txt> [<photo2> <truth2.txt> ...] [--stand-in]
"""
import argparse
import itertools
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import photo_scanner as ps


def get_words(text: str) -> set:
    """Extract lowercase words of length >= 3."""
    return set(re.findall(r"[a-z]{3,}", text.lower()))


class StandInTesseractReader:
    """
    Stand-in OCR reader wrapping Tesseract (or fallback simulation if Tesseract binary is absent)
    that accepts the same call shape as EasyOCR's readtext.
    Validates script logic without requiring EasyOCR.
    """
    def __init__(self):
        self.tess_ok = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tess_ok = True
        except Exception:
            self.tess_ok = False

    def readtext(self, img: np.ndarray, **kwargs) -> list:
        if self.tess_ok:
            import pytesseract
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            data = pytesseract.image_to_data(gray, lang="eng", output_type=pytesseract.Output.DICT)
            results = []
            for i, txt in enumerate(data["text"]):
                t = str(txt).strip()
                try:
                    conf = float(data["conf"][i]) / 100.0
                except (ValueError, TypeError):
                    continue
                if not t or conf <= 0:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                results.append((bbox, t, conf))
            return results

        # Fallback simulation for environments without Tesseract binary
        # Returns synthetic layout boxes from the image dimensions so pipeline runs end-to-end
        h, w = img.shape[:2]
        return [
            ([[50, 40], [w - 50, 40], [w - 50, 70], [50, 70]], "INGREDIENTS: Sugar, wheat flour, skimmed milk powder", 0.90),
            ([[50, 80], [w - 50, 80], [w - 50, 110], [50, 110]], "hazelnuts (13%), cocoa butter, soy lecithin, vanillin", 0.88),
            ([[50, 120], [w - 50, 120], [w - 50, 150], [50, 150]], "May contain traces of peanuts and sesame seeds.", 0.85),
        ]


def run_tuning(pairs: List[Tuple[str, str]], use_stand_in: bool = False) -> None:
    # Validate input pairs
    valid_pairs: List[Tuple[Path, str, set]] = []
    for img_arg, txt_arg in pairs:
        img_p, txt_p = Path(img_arg), Path(txt_arg)
        if not img_p.exists():
            print(f"Error: Photo '{img_p}' does not exist.")
            sys.exit(1)
        if not txt_p.exists():
            print(f"Error: Ground truth text file '{txt_p}' does not exist.")
            sys.exit(1)
        truth_text = txt_p.read_text(encoding="utf-8")
        true_words = get_words(truth_text)
        if not true_words:
            print(f"Warning: No 3+ letter words found in '{txt_p}'.")
        valid_pairs.append((img_p, truth_text, true_words))

    print(f"Loaded {len(valid_pairs)} photo/ground-truth pairs.")

    if use_stand_in:
        print("Using Stand-in reader (validating script logic without EasyOCR)...")
        reader = StandInTesseractReader()
    else:
        print("Initializing EasyOCR reader...")
        reader = ps.get_ocr_reader()

    # Pre-compute the pipeline base images for each photo (the exact image the app gives EasyOCR)
    print("\nPre-processing photos with normal pipeline...")
    photo_data: List[Dict[str, Any]] = []
    for img_p, truth_text, true_words in valid_pairs:
        t0 = time.time()
        pipe = ps.run_ocr_pipeline(str(img_p))
        base_img = pipe["image"]
        dt = round(time.time() - t0, 2)
        print(f"  Photo: {img_p.name} -> Base image shape: {base_img.shape} in {dt}s")
        photo_data.append({
            "path": img_p,
            "name": img_p.name,
            "base_img": base_img,
            "truth": truth_text,
            "true_words": true_words,
        })

    # Grid definition:
    # 1. Colour vs contrast-enhanced (clahe)
    # 2. Extra zoom: 1.0 vs 1.5
    # 3. add_margin: 0.03, 0.08, 0.15
    # 4. text_threshold: 0.4, 0.6
    # 5. decoder: greedy vs beamsearch
    grid_variants = ["color", "clahe"]
    grid_zooms = [1.0, 1.5]
    grid_margins = [0.03, 0.08, 0.15]
    grid_thresholds = [0.4, 0.6]
    grid_decoders = ["greedy", "beamsearch"]

    param_combinations = list(itertools.product(
        grid_variants, grid_zooms, grid_margins, grid_thresholds, grid_decoders
    ))
    total_combos = len(param_combinations)
    print(f"\nTesting {total_combos} parameter combinations on {len(photo_data)} photos ({total_combos * len(photo_data)} total evaluations)...")

    results: List[Dict[str, Any]] = []

    for idx, (variant, zoom, margin, threshold, decoder) in enumerate(param_combinations, start=1):
        if idx % 8 == 1 or idx == total_combos:
            print(f"  Progress: {idx}/{total_combos} combinations tested...")

        recalls: List[float] = []
        durations: List[float] = []

        for pinfo in photo_data:
            base_img = pinfo["base_img"]
            # 1. Variant
            v_img = base_img if variant == "color" else ps.clahe_gray(base_img)

            # 2. Extra Zoom
            if zoom != 1.0:
                h, w = v_img.shape[:2]
                z_img = cv2.resize(v_img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            else:
                z_img = v_img

            h_z, w_z = z_img.shape[:2]

            t_eval_start = time.time()
            if use_stand_in:
                raw_boxes = reader.readtext(z_img)
            else:
                raw_boxes = reader.readtext(
                    z_img,
                    paragraph=False,
                    mag_ratio=1.0,
                    canvas_size=int(max(h_z, w_z)),
                    add_margin=margin,
                    text_threshold=threshold,
                    decoder=decoder,
                )
            eval_time = time.time() - t_eval_start
            durations.append(eval_time)

            # Convert to OCRBoxes and filter min conf
            boxes = []
            for bbox, text, conf in raw_boxes:
                t = str(text).strip()
                if t and float(conf) >= ps.CONFIG["min_box_conf"]:
                    boxes.append(ps.OCRBox([[float(x), float(y)] for x, y in bbox], t, float(conf)))

            # Build text using normal scanner pipeline
            healed_text, ing_text = ps._build_text(boxes, heal=True)
            pred_text = ing_text if ing_text else healed_text
            pred_words = get_words(pred_text)

            true_words = pinfo["true_words"]
            if true_words:
                recall = len(true_words & pred_words) / len(true_words)
            else:
                recall = 1.0 if not pred_words else 0.0

            recalls.append(recall)

        mean_rec = statistics.mean(recalls)
        worst_rec = min(recalls)
        mean_sec = statistics.mean(durations)

        results.append({
            "variant": variant,
            "zoom": zoom,
            "add_margin": margin,
            "text_threshold": threshold,
            "decoder": decoder,
            "mean_recall": mean_rec,
            "worst_recall": worst_rec,
            "mean_seconds": mean_sec,
            "recalls": recalls,
        })

    # Order by: mean recall (descending), worst-photo recall (descending), speed/mean_seconds (ascending)
    results.sort(key=lambda r: (-r["mean_recall"], -r["worst_recall"], r["mean_seconds"]))

    print("\n" + "=" * 95)
    print("TOP 8 EASYOCR SETTINGS ORDERED BY MEAN RECALL, WORST-PHOTO RECALL, SPEED:")
    print("=" * 95)
    header = f"{'Rank':<5} {'Variant':<8} {'Zoom':<6} {'Margin':<8} {'Thresh':<8} {'Decoder':<12} {'Mean Rec':<10} {'Worst Rec':<10} {'Avg Sec':<8}"
    print(header)
    print("-" * 95)

    for rank, r in enumerate(results[:8], start=1):
        row = (f"{rank:<5} {r['variant']:<8} {r['zoom']:<6.1f} {r['add_margin']:<8.2f} "
               f"{r['text_threshold']:<8.2f} {r['decoder']:<12} "
               f"{r['mean_recall']:<10.3f} {r['worst_recall']:<10.3f} {r['mean_seconds']:<8.2f}s")
        print(row)
    print("=" * 95)

    winner = results[0]
    print("\nRECOMMENDED EASYOCR CONFIGURATION:")
    print(f"  variant:        {winner['variant']}")
    print(f"  extra_zoom:     {winner['zoom']}")
    print(f"  add_margin:     {winner['add_margin']}")
    print(f"  text_threshold: {winner['text_threshold']}")
    print(f"  decoder:        {winner['decoder']}")
    print(f"  mean_recall:    {winner['mean_recall']:.3%}")
    print(f"  worst_recall:   {winner['worst_recall']:.3%}")
    print(f"  mean_speed:     {winner['mean_seconds']:.2f}s per photo")


def main():
    parser = argparse.ArgumentParser(description="EasyOCR Hyperparameter Grid Tuning")
    parser.add_argument("args", nargs="+", help="Pairs of <photo> <truth.txt>")
    parser.add_argument("--stand-in", action="store_true", help="Use stand-in reader to validate script without EasyOCR")
    parsed = parser.parse_args()

    items = parsed.args
    if len(items) % 2 != 0:
        print("Error: Arguments must be provided in pairs: <photo> <truth.txt> [<photo2> <truth2.txt> ...]")
        sys.exit(1)

    pairs = list(zip(items[0::2], items[1::2]))
    run_tuning(pairs, use_stand_in=parsed.stand_in)


if __name__ == "__main__":
    main()
