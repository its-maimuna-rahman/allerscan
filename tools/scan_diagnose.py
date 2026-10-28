#!/usr/bin/env python3
"""
Diagnostic tool for AllerScan photo scanning.
Inspects each stage of the OCR pipeline on a given photo without modifying the app.

Usage:
    python scan_diagnose.py <photo_path> [--engine {easyocr,tesseract,both}]
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np

import photo_scanner as ps


def run_diagnostics(photo_path: str, engine: Optional[str] = None) -> None:
    p = Path(photo_path)
    if not p.exists():
        print(f"Error: Photo not found at '{photo_path}'")
        sys.exit(1)

    print("=" * 70)
    print(f"ALLERSCAN OCR DIAGNOSTICS: {p.name}")
    print("=" * 70)

    try:
        pipe = ps.run_ocr_pipeline(str(p), engine=engine)
    except Exception as exc:
        print(f"Fatal error running OCR pipeline: {exc}")
        sys.exit(1)

    # 1. Quality metrics, rotation, skew, notes, timings, final image size
    print("\n--- 1. PHOTO QUALITY & GEOMETRY ---")
    q = pipe.get("quality", {})
    w, h = q.get("width"), q.get("height")
    print(f"Input dimensions:  {w}x{h} px")
    print(f"Sharpness (blur):  {q.get('blur_var')} (var of Laplacian)")
    print(f"Brightness:        {q.get('brightness')} (0-255)")
    print(f"Contrast:          {q.get('contrast')} (std dev)")
    print(f"Glare fraction:    {q.get('glare_fraction')}")
    if q.get("warnings"):
        print("Quality warnings:")
        for warn in q["warnings"]:
            print(f"  [!] {warn}")
    else:
        print("Quality warnings:  None")

    print(f"Rotation applied:  {pipe.get('angle', 0)}°")
    print(f"Skew applied:      {pipe.get('skew', 0.0):+.2f}°")
    final_img = pipe.get("image")
    if final_img is not None:
        print(f"Final image size:  {final_img.shape[1]}x{final_img.shape[0]} px (shape: {final_img.shape})")
    if pipe.get("notes"):
        print("Pipeline notes:    " + "; ".join(pipe["notes"]))
    timings = pipe.get("timings", {})
    if timings:
        print("Stage timings:     " + ", ".join(f"{k}={v}s" for k, v in timings.items()))

    # 2. Summary line per engine reading
    print("\n--- 2. ENGINE READINGS SUMMARY ---")
    candidates = pipe.get("candidates", [])
    if not candidates:
        print("No engine readings available.")
    for cand in candidates:
        eng_name = cand.get("engine", "unknown")
        variant = cand.get("variant", "unknown")
        n_boxes = len(cand.get("boxes", []))
        readab = cand.get("readability", {})
        v_words = readab.get("vocab_words", 0)
        m_conf = readab.get("mean_conf", 0.0)
        score = readab.get("score", 0.0)
        chosen = " [CHOSEN]" if cand is pipe.get("best") else ""
        print(f"* Engine '{eng_name}' ({variant}): {n_boxes} boxes | "
              f"{v_words} recognised food/label words | "
              f"mean conf: {m_conf:.2f} | score: {score:.1f}{chosen}")

    chosen_boxes: List[ps.OCRBox] = pipe.get("boxes", [])
    if not chosen_boxes:
        print("\n[!] No text boxes were detected by any OCR reading.")
        print("\n--- 3. DETECTED BOXES ---\nNo boxes found.")
        print("\n--- 4. LAYOUT LINES ---\nNo lines formed.")
        print("\n--- 5. HEALED & FINAL INGREDIENT TEXT ---\nNo text extracted.")
        print("\n--- 6. ANNOTATED IMAGE ---\nSkipped: no boxes to draw.")
        return

    # 3. Every box of the chosen reading ordered top to bottom
    print(f"\n--- 3. DETECTED BOXES (CHOSEN READING: {len(chosen_boxes)} BOXES) ---")
    print(f"{'#':<4} {'Conf':<6} {'Height':<8} {'Position (x0,y0)-(x1,y1)':<28} {'Text'}")
    print("-" * 70)
    ordered_boxes = sorted(chosen_boxes, key=lambda b: (b.y_min, b.x_min))
    for idx, b in enumerate(ordered_boxes, start=1):
        pos_str = f"({b.x_min:.0f},{b.y_min:.0f})-({b.x_max:.0f},{b.y_max:.0f})"
        print(f"{idx:<4} {b.conf:<6.2f} {b.h:<8.1f} {pos_str:<28} '{b.text}'")

    # 4. Text lines after layout (reading order)
    print("\n--- 4. LAYOUT LINES (READING ORDER) ---")
    grouped_lines = ps.group_boxes_into_lines(chosen_boxes)
    for idx, line in enumerate(grouped_lines, start=1):
        line_text = " ".join(b.text for b in line)
        print(f"Line {idx:02d}: {line_text}")

    # 5. Healed text and final ingredient text
    print("\n--- 5. HEALED & FINAL INGREDIENT TEXT ---")
    log = []
    healed_text, final_ingredient_text = ps._build_text(chosen_boxes, heal=True, log=log)
    print("[HEALED MULTILINE TEXT]:")
    print(healed_text if healed_text else "(empty)")
    if log:
        print("\n[WORD REPAIRS APPLIED]:")
        for orig, healed in log:
            print(f"  {orig} -> {healed}")
    print("\n[FINAL EXTRACTED INGREDIENT TEXT]:")
    print(final_ingredient_text if final_ingredient_text else "(empty)")

    # 6. Picture file saved next to the photo with detected boxes drawn
    print("\n--- 6. SAVING ANNOTATED IMAGE ---")
    if final_img is not None:
        canvas = final_img.copy() if final_img.ndim == 3 else cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
        low_conf_thr = ps.CONFIG.get("low_conf_word", 0.45)
        for b in chosen_boxes:
            # Green for confident, Red for low confidence (OpenCV is BGR)
            color = (0, 200, 0) if b.conf >= low_conf_thr else (0, 0, 220)
            cv2.rectangle(canvas, (int(b.x_min), int(b.y_min)), (int(b.x_max), int(b.y_max)), color, 2)
            cv2.putText(canvas, f"{b.conf:.2f}", (int(b.x_min), max(12, int(b.y_min) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        out_annotated = p.parent / f"{p.stem}_annotated_boxes.jpg"
        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_annotated), bgr)
        print(f"Annotated photo saved to: {out_annotated}")
        print(f"  Green boxes: confidence >= {low_conf_thr}")
        print(f"  Red boxes:   confidence < {low_conf_thr}")
    else:
        print("Cannot save annotated image: final image is not available.")


def main():
    parser = argparse.ArgumentParser(description="AllerScan OCR Diagnostics")
    parser.add_argument("photo", help="Path to the packaging photo file")
    parser.add_argument("--engine", choices=["easyocr", "tesseract", "both"], default=None,
                        help="OCR engine override (default: from CONFIG)")
    args = parser.parse_args()
    run_diagnostics(args.photo, engine=args.engine)


if __name__ == "__main__":
    main()
