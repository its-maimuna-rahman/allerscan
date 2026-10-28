# Section 2 Reproduction Evidence: Before Spell-Correction Upgrade

**Baseline test suite status**: 78 passed tests across 4 test files.

## Reproduction Table

| Input (ingredient text) | Current Parser Found | Scanner Heal Output | Parser on Healed Text | Gap / Diagnosis |
|---|---|---|---|---|
| `sugar, mlik powder, peenut oil, whaet flour, soya lecitin, casien, glutten, sesamee seeds, shrimpp, hazlenuts, allmonds, cashw` | soybean | `sugar, milk powder, peanut oil, wheat flour, soya lecithin, casein, gluten, sesame seeds, shrimp, hazelnuts, almonds, cashw` | almond, chicken_egg, cow_milk, hazelnut, peanut, sesame_seed, shrimp, soybean, wheat | Only `['soybean']` found by parser. Scanner healed some words (`['almond', 'chicken_egg', 'cow_milk', 'hazelnut', 'peanut', 'sesame_seed', 'shrimp', 'soybean', 'wheat']`) but missed `mlik`, `peenut`, `whaet`, `casien`, `glutten` and blindly converted `cashw` -> `cashew` without 'possible' flag. |
| `peanuts, almonds, eggs, prawns, shrimps, soybeans, walnuts, cashews` | almond, cashew, chicken_egg, english_walnut, peanut, shrimp, soybean | `peanuts, almonds, eggs, prawns, shrimps, soybeans, walnuts, cashews` | almond, cashew, chicken_egg, english_walnut, peanut, shrimp, soybean | All 7 intended allergens found. Existing parser plural handling works. |
| `skimmedmilk powder, wheatflour, peanutbutter, eggwhite` | **None** | `skimmed milk powder, wheat flour, peanut butter, egg white` | chicken_egg, cow_milk, peanut, wheat | Parser found none (`[]`). Scanner split `wheat flour`, `skimmed milk`, `peanut butter`, `egg white` so parser found all 4 on healed. |
| `MILK;SOY.EGG/WHEAT` | chicken_egg, cow_milk, soybean, wheat | `MILK,SOY.EGG/WHEAT` | chicken_egg, cow_milk, soybean, wheat | Parser found all 4 (`['chicken_egg', 'cow_milk', 'soybean', 'wheat']`). Punctuation handling already splits on `;`, `.`, `/`. |
