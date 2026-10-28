# AllerScan Confusable Pairs and Protected Words Report

Generated as required by Step 4 of `04_spelling_correction.md`.

## 1. Safety Rule
> **Rule 1**: A real, correctly spelled non-allergen word must **never** be rewritten into an allergen.
> Tokens on the **Protected List** are immune to fuzzy allergen substitution across all modes.

**Total Protected Words in Knowledge Base**: 10901

## 2. Key Confusable Pairs within Edit Distance 1-2

| Allergen Word | Real English Word | Distance | Protection Status | Note / Spec Reference |
|---|---|---|---|---|
| `pear` | `pearl` | 1 | **PROTECTED** | Pear / Pearl (jewel / barley) |
| `pear` | `peer` | 1 | **PROTECTED** | Pear / Peer |
| `rice` | `price` | 1 | **PROTECTED** | Rice / Price (label meta) |
| `sole` | `soul` | 1 | **PROTECTED** | Sole fish / Soul |
| `corn` | `born` | 1 | **PROTECTED** | Corn / Born |
| `corn` | `horn` | 1 | **PROTECTED** | Corn / Horn |
| `nut` | `not` | 1 | **PROTECTED** | Nut / Not (negation) |
| `soy` | `say` | 1 | **PROTECTED** | Soy / Say (common verb) |
| `fish` | `dish` | 1 | **PROTECTED** | Fish / Dish (tableware/culinary) |
| `milk` | `mild` | 1 | **PROTECTED** | Milk / Mild (flavor descriptor) |
| `milk` | `mill` | 1 | **PROTECTED** | Milk / Mill (grinding equipment) |
| `wheat` | `sheet` | 1 | **PROTECTED** | Wheat / Sheet (packaging) |
| `date` | `dare` | 1 | **PROTECTED** | Date / Dare |
| `bean` | `been` | 1 | **PROTECTED** | Bean / Been (verb form) |
| `bean` | `beat` | 1 | **PROTECTED** | Bean / Beat (action/beet) |
| `egg` | `ego` | 1 | **PROTECTED** | Egg / Ego |
| `barley` | `barely` | 2 | **PROTECTED + POSSIBLE** | Misspelling 'barely' stays barely; flagged as possible barley in label |

### Automated Confusable Sample (Representative Target Allergens)

| Allergen Target | Confusable English Token | Distance | Category |
|---|---|---|---|
| `barley` | `bawled` | 2 | Allergen-English |
| `barley` | `parlay` | 2 | Allergen-English |
| `barley` | `barlow` | 2 | Allergen-English |
| `barley` | `paley` | 2 | Allergen-English |
| `barley` | `balled` | 2 | Allergen-English |
| `barley` | `balky` | 2 | Allergen-English |
| `barley` | `halley` | 2 | Allergen-English |
| `barley` | `barren` | 2 | Allergen-English |
| `barley` | `balmy` | 2 | Allergen-English |
| `barley` | `ballsy` | 2 | Allergen-English |
| `barley` | `carney` | 2 | Allergen-English |
| `barley` | `bramley` | 2 | Allergen-English |
| `barley` | `galley` | 2 | Allergen-English |
| `barley` | `morley` | 2 | Allergen-English |
| `barley` | `bradley` | 2 | Allergen-English |
| `barley` | `harlem` | 2 | Allergen-English |
| `barley` | `valley` | 2 | Allergen-English |
| `barley` | `barnes` | 2 | Allergen-English |
| `barley` | `barges` | 2 | Allergen-English |
| `barley` | `bale` | 2 | Allergen-English |
| `barley` | `hurley` | 2 | Allergen-English |
| `barley` | `barer` | 2 | Allergen-English |
| `barley` | `bares` | 2 | Allergen-English |
| `barley` | `alley` | 2 | Allergen-English |
| `barley` | `blarney` | 2 | Allergen-English |
| `barley` | `garvey` | 2 | Allergen-English |
| `barley` | `beasley` | 2 | Allergen-English |
| `barley` | `earle` | 2 | Allergen-English |
| `barley` | `marley` | 1 | Allergen-English |
| `barley` | `bared` | 2 | Allergen-English |
| `barley` | `barker` | 2 | Allergen-English |
| `barley` | `parsley` | 2 | Allergen-Allergen |
| `barley` | `barked` | 2 | Allergen-English |
| `barley` | `barclay` | 2 | Allergen-English |
| `barley` | `bailed` | 2 | Allergen-English |
| `barley` | `parleys` | 2 | Allergen-English |
| `barley` | `burly` | 2 | Allergen-English |
| `barley` | `barged` | 2 | Allergen-English |
| `barley` | `barfed` | 2 | Allergen-English |
| `barley` | `barneys` | 2 | Allergen-English |
| `barley` | `barrel` | 2 | Allergen-English |
| `barley` | `baled` | 2 | Allergen-English |
| `barley` | `parley` | 1 | Allergen-English |
| `barley` | `barkley` | 1 | Allergen-English |
| `barley` | `barely` | 2 | Allergen-English |
| `barley` | `badly` | 2 | Allergen-English |
| `barley` | `manley` | 2 | Allergen-English |
| `barley` | `oakley` | 2 | Allergen-English |
| `barley` | `varlet` | 2 | Allergen-English |
| `barley` | `bailey` | 1 | Allergen-English |
| `barley` | `daley` | 2 | Allergen-English |
| `barley` | `berle` | 2 | Allergen-English |
| `barley` | `bales` | 2 | Allergen-English |
| `barley` | `talley` | 2 | Allergen-English |
| `barley` | `barney` | 1 | Allergen-English |
| `barley` | `farley` | 1 | Allergen-English |
| `barley` | `early` | 2 | Allergen-English |
| `barley` | `barter` | 2 | Allergen-English |
| `barley` | `bare` | 2 | Allergen-English |
| `barley` | `barber` | 2 | Allergen-English |

## 3. Ambiguous and Exact-Inflection Rules
- `oat` vs `oats`: Handled by exact inflection, never fuzzy mapping.
- `pea` vs `peas`: Handled by exact inflection.
- Words treated as ambiguous by parser (`sole`, `date`, `perch`, `bass`, `ray`, `rays`) are preserved without single-token allergen forcing.
- `barely`: Never rewritten to `barley`. If found in ingredient context, flagged as possible barley.
