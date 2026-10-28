# AllerScan photo scanning: instructions for the coding agent

You are improving the photo scanning and photo upload feature of AllerScan, a Streamlit food-allergy app. Follow the steps in order. Do not skip the verification at the end of each step. Write the code yourself; this file tells you what must be true, not how to type it.

---

## 1. Non-negotiable rules

1. **Safety first: recall over precision.** Never silently lose an allergen. Text may only be removed when it is clearly not food (nutrition numbers, lot codes, addresses, recycling text, web addresses). If you are unsure whether text matters, keep it.
2. **Never discard allergen statements.** "Contains: ...", "May contain ...", "Traces of ...", "Made in a facility that also processes ..." and "Allergy advice" sentences must always survive cleanup.
3. **Do not modify** the trained model and its artifacts, `model.py`, `train_model.py`, `generate_dataset.py`, `allergen_kb.py`, `name_mapping.json`, `nlp_parser.py`, `cross_reactivity.py`, `pipeline.py`. You may only read them.
4. **Keep public interfaces unchanged:** the functions `scan_photo` and `merge_scan_results` in `photo_scanner.py` keep their names, parameters and the result keys that `app.py` already reads. You may add new result keys. Do not remove any.
5. **Make small, separate changes.** One logical change per step, run the tests after each, and stop and report if something fails that you cannot explain.
6. **Do not invent results.** If you cannot run EasyOCR, Tesseract or a real photo, say so in your report instead of claiming success.
7. **Ask the human** for anything you cannot do yourself: installing system packages that need administrator rights, providing original photos, typing the true ingredient text of a pack.
8. Work on a new git branch and keep the previous versions of `photo_scanner.py` and `app.py` as backup copies before editing.

---

## 2. Background you must understand first

The project folder holds a Streamlit app (`app.py`), a photo scanner (`photo_scanner.py`), a benchmark tool (`scan_benchmark.py`), unit tests (`test_scanner_units.py`), an optional synthetic-photo generator (`make_synthetic.py`) and the existing allergen knowledge modules. The scanner reads a photo, finds the text, rebuilds the reading order, repairs OCR typos, extracts the ingredient list, and `app.py` then analyses it.

What the human reported, with two screenshots:

- **Webcam photo of a phone screen:** the Ingredients box contained only a heading sentence ("Understanding food ingredient lists for healthier choices") instead of the long ingredient list.
- **Uploaded photo of a Nutella jar:** most of the ingredient list was missing and junk appeared ("1 E 0"). The app also warned the photo was low resolution, which was correct (longest side under 900 pixels).
- **Both screenshots:** a stale "Matching products: Nutella" dropdown from an earlier search was still shown, although these were ingredient scans.

Established causes:

- **Proven bug in the extraction step.** The function that extracts the ingredient section finds the first match of an "ingredient" word and throws away all text before it. OCR had read the real header as "NGREDIENTS:" (first letter lost), which the header pattern did not match. A stray word "INGREDIENTS" at the end of the last line was matched instead, so the whole list before it was discarded and only trailing heading text survived.
- **Likely, not proven:** EasyOCR detects too few lines and misreads bold condensed or small print. Evidence: running the same pipeline with Tesseract on crops of both photos recovered almost the whole list for the first photo and every line except one bold line for the second. The two engines miss different lines, so using both is the planned remedy. You cannot confirm EasyOCR's behaviour unless EasyOCR runs on the human's machine, so measurement steps are included.
- **Proven UI bug.** The ingredient-scan result handler never clears old product matches, so the stale dropdown stays visible, and picking anything in it overwrites the ingredient text.

---

## 3. Steps

### Step 0. Prepare

- Create a new git branch named for this work.
- Save backup copies of `photo_scanner.py` and `app.py` with clear "before" names.
- Read `photo_scanner.py`, `app.py`, `test_scanner_units.py` and `scan_benchmark.py` fully before changing anything.
- Run the existing unit tests and record how many pass. The expected baseline is 20 passing. If the baseline fails, stop and report; do not continue.

### Step 1. Add a diagnostic tool (read-only behaviour)

Create a small standalone command-line script, separate from the app, that takes one photo path and an optional engine name (EasyOCR, Tesseract or both) and prints what each stage did:

1. Photo quality metrics and warnings, rotation applied, skew applied, pipeline notes, timings and the final image size.
2. One summary line per engine reading: number of boxes, count of recognised food/label words, mean confidence, score.
3. Every box of the chosen reading with confidence, height, position and text, ordered top to bottom.
4. The text lines after layout (reading order).
5. The healed text and the final ingredient text.
6. A picture file saved next to the photo with detected boxes drawn on it, green for confident and red for low confidence.

Use the scanner's existing internal pieces; do not duplicate its logic. If no text boxes are found, print a clear message instead of failing.

**Done when:** running it on a synthetic photo from the generator prints all six sections without errors.

Ask the human to run it on their two original photos (the Nutella jar photo and the phone-screen photo) and keep the output. This output is the "before" evidence. Do not continue past Step 8 without telling the human it exists, but you may continue coding while waiting.

### Step 2. Fix the header bug in the extraction step

Required behaviour:

1. The header pattern must also recognise an ingredients header whose first letter was lost by OCR (for example "NGREDIENTS"), in addition to the normal and the common letter-confusion variants it already handles.
2. Create a second, stricter notion called a **confirmed header**: an ingredients word that ends with a colon and is followed, within roughly the next 250 characters, by at least one comma. Only a confirmed header may be used to cut off text before it.
3. The function that picks a header must return the first confirmed header, or nothing. It must no longer fall back to "the first match of any kind".
4. If there is no confirmed header, **nothing is cut off**; the whole text is used, and only the clearly-non-food noise removal applies.
5. The function that classifies a scan as ingredient list versus product front must use the confirmed-header notion, not the loose one, so marketing phrases such as "made with natural ingredients" or "ingredient lists for healthier choices" no longer count as an ingredient list. A "Contains:" statement still counts as before.
6. Allergen statements are still collected from anywhere in the text and appended after the ingredient body, exactly as before.

Also add a handful of extra harmless words to the list of words that typo-healing must never alter: monoglyceride, monoglycerides, diglyceride, diglycerides, distilled. This prevents "MonoglyCeRIDES" being split into two words.

**Done when:** the behaviours in Step 3 pass.

### Step 3. Add regression tests

Add these tests to the existing unit test file. Each one is described by input and expected outcome:

1. **Lost first letter.** Several lines of an ingredient list starting with "NGREDIENTS:" followed by water, corn syrup, canola oil and a caseinate containing "(MILK)", then a line ending with "INGREDIENTS", then a heading sentence about "food ingredient lists for healthier choices". Expect: the extracted text still contains "CORN SYRUP", "CASEINATE" and "LECITHIN" from the list, and milk and soy are detected as allergens.
2. **Stray word.** A normal comma-separated list (sugar, wheat flour, milk powder, salt) followed by the sentence "See all INGREDIENTS on the back". Expect wheat and cow milk still detected.
3. **Heading sentence is not a header.** The text "Understanding food ingredient lists for healthier choices" followed by "Water, sugar, soy lecithin". Expect soy detected.
4. **Marketing word.** The text "Made with natural ingredients and real hazelnuts" must be classified as product front, not ingredient list.

**Done when:** all tests pass; the expected total is 24 (20 original plus these 4). Report the exact count.

### Step 4. Use both OCR engines by default

1. Change the default engine setting to "both", while keeping the environment variable override working.
2. Confirm the engine factory degrades gracefully: if only one engine is installed it must log a warning and use that one, and if neither is installed it must raise the existing clear installation message. Do not let a missing Tesseract crash the app.
3. Ask the human to install the Tesseract system package and its Python wrapper if they are not installed (this needs administrator rights, so do not attempt it silently). Verify with the diagnostic tool that its summary lists readings from both engines.
4. Do not change how the best reading is chosen or how the allergen union ("uncertain allergens") works.

**Done when:** with both engines installed the diagnostic output lists readings from both; with one uninstalled the scan still works.

### Step 5. Make EasyOCR settings tunable, then tune them with data

1. In the EasyOCR engine's read function, pass every entry of the EasyOCR section of the configuration dictionary straight through to EasyOCR's text-reading call, in addition to the fixed arguments it already sets (no paragraph mode, magnification ratio 1.0, canvas size equal to the longer image side). This lets extra options such as the decoder type be set from configuration without further code changes. Do not rename existing keys.
2. Create a standalone tuning script that takes pairs of arguments: a photo and a text file with its true text. For each photo it must first run the normal pipeline once to get the exact image the app would give EasyOCR, then try this grid on that image: colour versus contrast-enhanced variant, extra zoom of 1.0 versus 1.5, box margin of 0.03, 0.08 and 0.15, detection threshold of 0.4 and 0.6, and greedy versus beam-search decoding. For each combination it builds the text with the scanner's normal text-building step and scores **word recall**: the share of true words (three letters or more) that were found. It prints progress, then the top eight settings ordered by mean recall, then worst-photo recall, then speed, and finally the recommended values to put into the configuration.
3. Validate the script's logic without EasyOCR by testing it against a stand-in reader that wraps Tesseract and accepts the same call shape. Make clear in your report that this only proves the script runs, not that any setting is good.
4. Ask the human for the ground-truth text of their photos (they must type it from the real pack) and to run the script on their machine with EasyOCR installed. Apply the winning values to the configuration **only if** they improve the worst photo, not just the average. If no setting beats the defaults, leave the defaults and say that the photo, not the settings, is the limit.

**Done when:** the existing tests still pass and the tuning script runs end to end against the stand-in reader.

### Step 6. Remove the stale product dropdown after ingredient scans

In the app's scan-result handler, in the branch that handles an ingredient list, before writing the ingredient text: clear the stored list of product matches and remove the stored dropdown selection. Do not touch the front-of-pack branch, which legitimately fills these. Keep the order of operations so the ingredient text box shows the new text.

**Done when:** after an ingredient scan the product-match dropdown is not shown. Test the handler with a minimal fake of the Streamlit object if Streamlit is not installed; state that you could not launch the real UI.

### Step 7. Add an "Other readings" picker

1. In the scan function, after the engines have run, build a list of readings: one entry per engine-and-variant reading that found boxes, each with a label (engine/variant), the cleaned ingredient text built from its boxes, the number of boxes, its score, and a flag saying whether it is the chosen reading. Store it in the result under a new key named for readings. Place this before the existing code that stores corrections.
2. In the function that merges several photos, set the readings key to an empty list so a merged result never shows another photo's readings.
3. In the app's scan details expander, when there are two or more readings, show a radio selection of the readings (labels include box count, score and "(chosen)"), show the selected reading's text as plain read-only text, and a button "Use this reading in the ingredient box" which writes the selected text into the ingredient box state and marks the analysis as requested. Place this block **before** the ingredient text area is created in the script, because it changes that widget's state. Show the text with a display element that updates correctly when the selection changes, not a disabled text area with a fixed key. Do not use any rerun call; it is not needed and breaks older Streamlit versions.

**Done when:** a scan of a deliberately noisy synthetic image with both engines active produces at least two readings, and the app file compiles.

### Step 8. Guidance in the app for photo quality

Verify the existing photo tips in the scan tab say all of the following, and add any that are missing as short plain-language lines:

- Use the Upload option with the phone's own camera for small print; the browser webcam option gives much lower resolution.
- Do not photograph a phone or computer screen.
- Fill the frame with the ingredient list, tap to focus, use daylight, tilt the pack slightly to avoid glare.
- For curved cans or bottles, take two or three photos while turning the pack and upload all of them.
- Very small image files (longest side under 900 pixels) cannot be read reliably; a photo of a screen or a downloaded thumbnail is not a valid test.

Do not add image-processing code for this step.

### Step 9. Full verification

1. Run the whole unit test file. Expect 24 passing.
2. Compile-check every changed Python file.
3. Run the benchmark tool on the synthetic photos with the Tesseract engine and with both engines. Expect zero missed allergens and perfect precision and recall on this clean synthetic set; if not, investigate before reporting. Say clearly that synthetic results do not prove real-world accuracy.
4. Run the diagnostic tool on the two crops or photos available to you. For the phone-screen type of photo, confirm the final ingredient text now begins at the first ingredient after the header and contains the long list rather than the heading sentence.
5. Check by reading the code that no text is removed when no confirmed header exists, and that allergen statements are still appended.
6. Ask the human to repeat their two photos in the real app with EasyOCR and Tesseract installed, and to report: the number of text blocks, whether the ingredient box is complete, whether "Other readings" appears, and whether the stale dropdown is gone.

### Step 10. Report back

Give the human a short report with:

- which steps were completed, with the test counts before and after;
- which files you changed or created;
- what you could not verify (EasyOCR behaviour, real photos, the live Streamlit screen) and what the human must run;
- anything surprising, and any remaining failure with its evidence (the diagnostic output).

If after all steps the human's real photos are still poorly read, do not keep changing code blindly. Ask for: the original photo file (not a screenshot), the typed true text, the full diagnostic output with both engines, and the box picture. Then decide whether detection, recognition, layout or cleanup is failing and fix only that part. Possible later options, not to be started without the human's agreement, are a stronger OCR model (such as PaddleOCR) or a vision-language model as a fallback for hard packaging.

---

## 4. Definition of done

- Existing and new unit tests pass (24 expected).
- A list that has a damaged header is never discarded.
- Both OCR engines are used by default and degrade gracefully when one is missing.
- EasyOCR settings can be changed from configuration, and a tuning script exists to choose them from the human's own photos.
- Ingredient scans no longer leave a stale product dropdown.
- The user can switch between readings when one misses a line.
- No file in the "do not modify" list was changed, and public function names and result keys are unchanged.
- The report states honestly what was and was not verified.