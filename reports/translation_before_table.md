# Task 5 Step 1: Multilingual Baseline "Before" Evidence Table

**Date**: 2026-10-27  
**Branch**: `ingredient-translation`  
**Test Suite Baseline**: 126 passed tests across 5 test suites.

---

## 1. Baseline Evidence: Current Scanner on Multilingual Pack Images

Tested on rendered label packaging images containing standard allergen formulas:
- Formula: Milk / Milk powder, Peanut / Peanut oil, Wheat / Wheat flour, Sugar.

| Language / Script | Ground Truth Text | Current Scanner OCR Output | Current Parser Allergens Found | Outcome / Safety Failure |
|---|---|---|---|---|
| **Bengali (`bn`)** (Bengali script) | `উপকরণ: দুধ, চিনাবাদাম, গম, চিনি।` | `""` (empty string) | `[]` (none) | ❌ **Silent Miss**: Returns "success: True" and "no allergens detected" on milk, peanut, and wheat! |
| **Arabic (`ar`)** (Arabic script, RTL) | `المكونات: حليب، فول سوداني، قمح، سكر.` | `""` (empty string) | `[]` (none) | ❌ **Silent Miss**: Completely unread; false sense of safety. |
| **French (`fr`)** (Latin script with accents) | `Ingrédients: lait, arachides, blé, sucre.` | `""` (empty string, orientation/header rejected) | `[]` (none) | ❌ **Silent Miss**: Standard French allergens missed. |
| **Chinese (`zh`)** (Han characters, no spaces) | `配料：牛奶，花生，小麦，白糖。` | `""` (empty string) | `[]` (none) | ❌ **Silent Miss**: Chinese packaging unread; 0 allergens found. |

---

## 2. Baseline Evidence: Current Parser on Pasted Foreign Text

When a user pastes non-English ingredient text directly into the text box:

| Input Text | Language | Expected Allergens | Current Parser Result | Status |
|---|---|---|---|---|
| `উপকরণ: দুধ, চিনাবাদাম, গম, চিনি।` | Bengali | Milk (`cow_milk`), Peanut (`peanut`), Wheat (`wheat`) | `[]` | ❌ 100% Missed |
| `Ingrédients: lait, arachides, blé, sucre.` | French | Milk (`cow_milk`), Peanut (`peanut`), Wheat (`wheat`) | `[]` | ❌ 100% Missed |
| `المكونات: حليب، فول سوداني، قمح، سكر.` | Arabic | Milk (`cow_milk`), Peanut (`peanut`), Wheat (`wheat`) | `[]` | ❌ 100% Missed |
| `配料：牛奶，花生，小麦，白糖。` | Chinese | Milk (`cow_milk`), Peanut (`peanut`), Wheat (`wheat`) | `[]` | ❌ 100% Missed |
| `सामग्री: दूध, मूंगफली, गेहूं, चीनी।` | Hindi | Milk (`cow_milk`), Peanut (`peanut`), Wheat (`wheat`) | `[]` | ❌ 100% Missed |

---

## 3. Product Database Filter Evidence

In `app.py`:
```python
letters = [c for c in ing if c.isalpha()]
if letters and (sum(c.isascii() for c in letters) / len(letters)) < 0.85:
    continue  # non-Latin / non-English list
```
Products returned by Open Food Facts with non-Latin ingredient lists (e.g. products sold in Bangladesh, India, Japan, China, Arab countries, Russia) are discarded at line 214 and never shown to the user.

---

**Step 1 Gate Summary**:
- English assumptions fully catalogued across 8 subsystems.
- Baseline before table established with 100% silent failure rate across non-English packaging and pasted text.
