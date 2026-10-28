# Phase 1 Supported Languages, Scripts, Reader Groups, and Licences

Documenting decisions and evidence for Step 2 of `docs/05_ingredient_translation.md`.

---

## 1. Phase 1 Language Matrix

| Language | ISO Code | Script | Direction | Spacing | EasyOCR Lang Codes | Script Group | Tesseract Code | Translation Candidate Models |
|---|---|---|---|---|---|---|---|---|
| **English** | `en` | Latin | LTR | Spaced | `['en']` | `latin` | `eng` | Direct (No translation required) |
| **Bengali** | `bn` | Bengali | LTR | Spaced | `['bn', 'en']` | `bengali` | `ben` | Helsinki-NLP/opus-mt-bn-en, Multilingual Glossary |
| **Hindi** | `hi` | Devanagari | LTR | Spaced | `['hi', 'en']` | `devanagari` | `hin` | Helsinki-NLP/opus-mt-hi-en, Multilingual Glossary |
| **Urdu** | `ur` | Arabic (Nastaliq) | RTL | Spaced | `['ur', 'en']` | `arabic` | `urd` | Helsinki-NLP/opus-mt-ur-en, Multilingual Glossary |
| **Arabic** | `ar` | Arabic | RTL | Spaced | `['ar', 'en']` | `arabic` | `ara` | Helsinki-NLP/opus-mt-ar-en, Multilingual Glossary |
| **Indonesian** | `id` | Latin | LTR | Spaced | `['id', 'en']` | `latin` | `ind` | Helsinki-NLP/opus-mt-id-en, Multilingual Glossary |
| **Malay** | `ms` | Latin | LTR | Spaced | `['ms', 'en']` | `latin` | `msa` | Helsinki-NLP/opus-mt-ms-en, Multilingual Glossary |
| **Thai** | `th` | Thai | LTR | **No Spaces** | `['th', 'en']` | `thai` | `tha` | Helsinki-NLP/opus-mt-th-en, Multilingual Glossary |
| **Chinese (Simplified)**| `zh-CN` | Han (Simplified)| LTR | **No Spaces** | `['ch_sim', 'en']` | `chinese_sim` | `chi_sim` | Helsinki-NLP/opus-mt-zh-en, Multilingual Glossary |
| **Chinese (Traditional)**| `zh-TW`| Han (Traditional)| LTR | **No Spaces**| `['ch_tra', 'en']` | `chinese_tra` | `chi_tra` | Helsinki-NLP/opus-mt-zh-en, Multilingual Glossary |
| **Japanese** | `ja` | Kanji / Kana | LTR | **No Spaces** | `['ja', 'en']` | `japanese` | `jpn` | Helsinki-NLP/opus-mt-jap-en, Multilingual Glossary |
| **Korean** | `ko` | Hangul | LTR | Spaced | `['ko', 'en']` | `korean` | `kor` | Helsinki-NLP/opus-mt-ko-en, Multilingual Glossary |
| **French** | `fr` | Latin | LTR | Spaced | `['fr', 'en']` | `latin` | `fra` | Helsinki-NLP/opus-mt-fr-en, Multilingual Glossary |
| **German** | `de` | Latin | LTR | Spaced | `['de', 'en']` | `latin` | `deu` | Helsinki-NLP/opus-mt-de-en, Multilingual Glossary |
| **Spanish** | `es` | Latin | LTR | Spaced | `['es', 'en']` | `latin` | `spa` | Helsinki-NLP/opus-mt-es-en, Multilingual Glossary |
| **Italian** | `it` | Latin | LTR | Spaced | `['it', 'en']` | `latin` | `ita` | Helsinki-NLP/opus-mt-it-en, Multilingual Glossary |
| **Portuguese** | `pt` | Latin | LTR | Spaced | `['pt', 'en']` | `latin` | `por` | Helsinki-NLP/opus-mt-pt-en, Multilingual Glossary |
| **Turkish** | `tr` | Latin | LTR | Spaced | `['tr', 'en']` | `latin` | `tur` | Helsinki-NLP/opus-mt-tr-en, Multilingual Glossary |
| **Russian** | `ru` | Cyrillic | LTR | Spaced | `['ru', 'en']` | `cyrillic` | `rus` | Helsinki-NLP/opus-mt-ru-en, Multilingual Glossary |

---

## 2. Licences and Provenance Audit

| Resource | Version / Source | Licence | Use in AllerScan | Compatibility Notes |
|---|---|---|---|---|
| **EasyOCR** | 1.7.2 (JaidedAI) | **Apache 2.0** | Offline optical character recognition | Fully permissive; commercial & non-commercial permitted. |
| **Tesseract OCR** | 5.5.0 (Google/HP) | **Apache 2.0** | Secondary OCR fallback engine | Fully permissive. |
| **Open Food Facts Taxonomies** | Ingredients, allergens, additives exports | **ODbL v1.0 / DbCL v1.0** | Seed dataset for multilingual glossary | Open Database License; requires attribution and share-alike on database modifications. Cached offline. |
| **EU Regulation (EU) No 1169/2011** | Annex II official 24 EU language translations | **Public Sector Information** | Official EU 14 allergen terminology | Free reuse with EUR-Lex citation. |
| **Helsinki-NLP Opus-MT** | University of Helsinki (Tatoeba / OPUS) | **CC-BY-4.0** | Offline neural machine translation models | Attribution required. Fully open. |
| **Argos Translate** | Argos Open Technologies | **MIT** | Offline translation engine candidate | Fully permissive. |
| **Meta NLLB-200** | Meta AI | **CC-BY-NC 4.0** | Evaluated candidate | Non-commercial restriction; Opus-MT and Argos preferred for open flexibility. |

---

## 3. Translation Engine Architecture (Pluggable Interface)

To support Route B and satisfy Rule 4 (Offline first, swappable engines), we define an abstract translation interface:
```python
class BaseTranslationEngine:
    def translate_items(self, items: List[str], source_lang: str, target_lang: str = "en") -> TranslationBatchResult:
        ...
```
Supported implementations:
1. **Multilingual Glossary Direct Translator** (Rule 2 Route A & Fallback): Guarantees offline translation of culinary items, allergens, and additives even without network or heavy models.
2. **Offline MarianMT / Opus-MT Translator**: Neural model execution via lightweight PyTorch/Transformers pipeline.
3. **External / API Fallback Engine**: Optional endpoint supporting user-provided keys, disabled by default.
