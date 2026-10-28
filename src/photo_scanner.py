"""
Photo Scanner for AllerScan  (v3)

What this module does, in order:
  1. Loads the photo (EXIF-aware), measures quality (blur / brightness / size) and warns early.
  2. Reads barcodes (EAN/UPC, checksum-validated) - an exact product lookup beats OCR of the front.
  3. Finds the page orientation (0/90/180/270) and removes small skew, using OCR readability
     (allergen/ingredient vocabulary hits), not "number of boxes".
  4. Measures the real text height and re-scales the image so small print is read at a size the
     OCR engine handles well ("adaptive zoom"). Runs one or more OCR engines/variants.
  5. Rebuilds reading order (multi-column aware), conservatively heals OCR typos, extracts the
     ingredient list WITHOUT dropping allergen warnings.
  6. Cross-checks every engine/variant and returns "uncertain" allergens (seen by one reading but not
     in the chosen text) so a false negative is never silent.
  7. Front-of-pack: searches Open Food Facts and only accepts a product whose name/brand really
     matches the text that was read.

Public entry points used by app.py:
  scan_photo(image_input, mode="auto", search_fn=None, engine=None) -> dict
  merge_scan_results(list_of_scan_dicts) -> dict
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from rapidfuzz import fuzz, process
from collections import OrderedDict

import allergen_kb as kb
from nlp_parser import HIDDEN_KB, parse_label
import spelling_corrector

try:
    from .multilingual_parser import (
        SCRIPT_RANGES,
        detect_language,
        detect_script,
        get_glossary_matcher,
        normalize_multilingual_text,
    )
    from .multilingual_fusion import (
        MultilingualAnalysisResult,
        VerificationState,
        analyze_multilingual_label,
    )
except ImportError:
    from multilingual_parser import (
        SCRIPT_RANGES,
        detect_language,
        detect_script,
        get_glossary_matcher,
        normalize_multilingual_text,
    )
    from multilingual_fusion import (
        MultilingualAnalysisResult,
        VerificationState,
        analyze_multilingual_label,
    )

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent

# -----------------------------------------------------------------------------
# Tunable settings (the benchmark script can override these with environment variables)
# -----------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    # "easyocr", "tesseract" or "both" (both = two independent readings, allergens are unioned)
    "engine": os.environ.get("ALLERSCAN_OCR_ENGINE", "both").lower(),
    # "auto": plain RGB first, add a contrast-enhanced reading only when the first one is weak
    # "rgb" / "clahe": force one variant (used by the benchmark sweep)
    "variant_mode": os.environ.get("ALLERSCAN_VARIANT", "auto").lower(),
    "pass1_long_side": 1920,        # first (fast) look at the photo
    "max_input_long_side": 4096,    # never keep more than this from the original photo
    "max_pass2_long_side": 3200,    # upper limit of the zoomed second look
    "max_total_upscale": 3.0,       # never enlarge the original photo more than this
    "target_box_height": int(os.environ.get("ALLERSCAN_TARGET_H", "32")),  # px height of a text line box
    "min_box_conf": 0.20,           # drop OCR boxes below this confidence
    "low_conf_word": 0.45,          # words below this are listed as "please verify"
    "skew_min_deg": 0.7,            # smaller skew is ignored
    "skew_search_deg": 12.0,        # search range +/- degrees
    "skew_min_gain": 1.15,          # profile-variance gain needed to accept a skew correction
    "easyocr": {                    # passed to EasyOCR readtext (names are EasyOCR's own)
        "text_threshold": 0.5, "low_text": 0.3, "link_threshold": 0.35, "add_margin": 0.15,
        "contrast_ths": 0.1, "adjust_contrast": 0.5, "min_size": 10,
    },
    "tesseract_psm": int(os.environ.get("ALLERSCAN_TESS_PSM", "11")),
    "match_min_score": 60,          # front-of-pack: min match score (0-100) to accept an Open Food Facts hit
}

# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------
COMMON_FOODS = {
    "wheat", "flour", "semolina", "durum", "starch", "cornstarch", "tapioca",
    "potato", "rice", "oat", "oats", "barley", "rye", "gluten", "malt", "bran",
    "buckwheat", "quinoa", "millet", "spelt", "sorghum",
    "sugar", "cane", "beet", "glucose", "fructose", "dextrose", "sucrose",
    "maltose", "lactose", "corn", "syrup", "molasses", "honey", "caramel",
    "invert", "sorbitol", "xylitol", "maltitol", "sucralose", "aspartame", "isomalt",
    "oil", "palm", "sunflower", "canola", "rapeseed", "olive", "coconut",
    "soybean", "cottonseed", "safflower", "peanut", "sesame", "butter",
    "margarine", "shortening", "cocoa", "lard", "tallow", "fat", "fats", "lipids", "shea",
    "milk", "skimmed", "whole", "cream", "whey", "casein", "caseinate",
    "cheese", "cheddar", "yogurt", "curd", "buttermilk", "ghee", "dairy",
    "egg", "eggs", "yolk", "albumen", "ovalbumin",
    "hazelnut", "hazelnuts", "almond", "almonds", "cashew", "cashews",
    "walnut", "walnuts", "pecan", "pecans", "pistachio", "pistachios",
    "macadamia", "peanuts", "chia", "flax", "flaxseed",
    "pumpkin", "poppy", "seeds", "seed",
    "soy", "soya", "soybeans", "lecithin", "pea", "peas", "lentil", "lentils",
    "chickpea", "chickpeas", "bean", "beans", "tomato", "onion", "garlic",
    "carrot", "celery", "mustard",
    "fish", "salmon", "tuna", "cod", "anchovy", "pollock", "crustacean",
    "crustaceans", "shrimp", "prawn", "crab", "lobster", "mollusc", "molluscs", "squid", "oyster",
    "vanilla", "vanillin", "yeast", "extract", "natural", "artificial",
    "flavor", "flavour", "flavors", "flavours", "flavoring", "flavouring",
    "spice", "spices", "preservative", "preservatives", "emulsifier", "emulsifiers",
    "protein", "fiber", "powder", "water", "salt", "sea",
    "citric", "acid", "potassium", "sorbate", "sodium", "bicarbonate",
    "carbonate", "concentrate", "raising", "agent", "agents", "pectin",
    "gelatin", "carrageenan", "xanthan", "gum", "guar", "dextrin",
    "maltodextrin", "chloride", "phosphate", "sulfite", "sulfites",
    "antioxidant", "ascorbic", "tocopherol", "tocopherols",
}

STRUCTURAL_VOCAB = {
    "ingredients", "ingredient", "contains", "contain", "may", "might", "could",
    "traces", "trace", "produced", "processed", "made", "manufactured", "facility",
    "equipment", "factory", "line", "allergen", "allergens", "advice", "warning",
    "information", "and", "or", "of", "with", "in", "from", "free", "also", "handles",
}

# Ordinary words that appear on packs. They must NEVER be "healed" into something else.
ENGLISH_COMMON = {
    "clean", "cool", "dry", "place", "keep", "store", "best", "before", "date",
    "see", "pack", "batch", "lot", "number", "fresh", "box", "lid", "cap",
    "bottom", "side", "top", "directions", "use", "serving", "size", "daily",
    "value", "energy", "net", "weight", "grams", "quality", "product", "recycle",
    "call", "visit", "website", "great", "crispy", "crunchy", "sweet", "rich",
    "pure", "choice", "premium", "classic", "original", "recipe", "roasted",
    "toasted", "dried", "organic", "solid", "solids", "mass", "paste", "pieces",
    "dark", "white", "light", "colour", "color", "colours", "colors", "clove",
}
LABEL_SAFE_WORDS = {
    "acidity", "regulator", "regulators", "stabiliser", "stabilisers", "stabilizer", "stabilizers",
    "thickener", "thickeners", "ginger", "barely", "pearl", "pearls", "cumin", "turmeric", "paprika",
    "coriander", "cinnamon", "pepper", "peppers", "chilli", "chili", "vinegar", "maize", "seasoning",
    "seasonings", "sweetener", "sweeteners", "colouring", "colourings", "coloring", "colorings",
    "flavourings", "vitamin", "vitamins", "riboflavin", "thiamin", "thiamine", "niacin", "folic",
    "iron", "calcium", "zinc", "magnesium", "ferment", "cultures", "culture", "enzymes", "enzyme",
    "rennet", "glaze", "glazing", "filling", "fillings", "crumb", "crumbs", "topping", "sauce",
    "sauces", "stock", "broth", "puree", "pulp", "juice", "juices", "peel", "zest", "mint", "basil",
    "oregano", "thyme", "rosemary", "sage", "essence", "aroma", "aromas", "wafer", "wafers",
    "biscuit", "biscuits", "cookie", "cookies", "cake", "cakes", "bread", "breadcrumbs", "pasta",
    "noodles", "vermicelli", "chocolate", "nutmeg", "mace", "cloves", "anise", "saffron", "tamarind",
    "jaggery", "raisin", "raisins", "sultana", "sultanas", "currant", "currants", "lemon", "lime",
    "orange", "wholegrain", "wholemeal", "sugars", "salts", "starches", "stabilised", "modified",
    "hydrolysed", "hydrolyzed", "vegetable", "vegetables", "fruit", "fruits", "bulking", "gelling",
    "raising", "glucose", "tartrate", "tartaric", "lactic", "malic", "acetic", "benzoate", "nitrite",
    "nitrate", "sulphite", "sulphites", "sulphur", "dioxide", "annatto", "carotene", "curcumin",
    "cochineal", "lycopene", "riboflavins", "polysorbate", "lecithins", "mono", "diglycerides",
    "monoglyceride", "monoglycerides", "diglyceride", "distilled",
    "glycerides", "glycerol", "cellulose", "silicon", "silicone", "talc", "beeswax", "wax", "shellac",
    "tapioca", "arrowroot", "cornflour", "cornmeal", "oatmeal", "wheatgerm", "germ", "husk", "husks",
    "flakes", "puffs", "granules", "crystals", "inulin", "chicory", "stevia", "steviol", "glycosides",
    "erythritol", "maltitol", "mannitol", "sorbitol", "acesulfame", "saccharin", "cyclamate",
    "fortified", "enriched", "bleached", "unbleached", "pasteurised", "pasteurized", "homogenised",
    "concentrated", "reconstituted", "dehydrated", "powdered", "ground", "crushed", "minced", "diced",
    "sliced", "smoked", "cured", "salted", "unsalted", "sweetened", "unsweetened", "flavoured",
    "flavored", "seasoned", "marinade", "dressing", "mayonnaise", "ketchup", "relish", "pickle",
    "pickled", "gherkin", "capers", "olives", "berries", "berry", "cherries", "grapes", "dates",
}

STEM_EXPANSIONS = {
    "mk": "milk", "mlk": "milk", "mik": "milk",
    "wey": "whey", "wney": "whey",
    "flou": "flour", "fiout": "flour", "flout": "flour",
    "pwder": "powder", "powdr": "powder", "poider": "powder",
    "lecitin": "lecithin", "lecithn": "lecithin", "lecith": "lecithin",
    "coco": "cocoa", "cacoa": "cocoa", "cacao": "cocoa",
    "buttel": "butter", "buttr": "butter",
    "hazel": "hazelnut", "hazelnu": "hazelnut",
    "eanut": "peanut", "peanu": "peanut",
    "almon": "almond", "lmond": "almond", "lmonds": "almonds",
    "cashe": "cashew", "walnu": "walnut", "pistach": "pistachio",
    "sesam": "sesame", "mustar": "mustard", "celer": "celery",
    "crustac": "crustacean", "mollus": "mollusc", "sy": "soy",
}

# first letters OCR commonly swaps; used to allow a fuzzy match when the first letter is damaged
CONFUSABLE_FIRST = {"m": "n", "n": "m", "c": "eo", "e": "c", "o": "c", "l": "i", "i": "l",
                    "b": "h", "h": "b", "u": "v", "v": "uw", "w": "v"}


def _load_allergen_vocab() -> Tuple[set, list, set, set, set]:
    """phrases, sorted tokens, single-word allergen words, full token vocab, fuzzy targets."""
    phrases: set = set()
    tokens: set = set()
    singles: set = set()
    targets: set = set()
    mapping_path = BASE_DIR / "name_mapping.json"
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for canonical, syns in data.items():
                for term in [canonical, *syns]:
                    clean = term.replace("_", " ").lower()
                    phrases.add(clean)
                    parts = clean.split()
                    tokens.update(parts)
                    if len(parts) == 1:
                        singles.add(clean)
                        if len(clean) >= 5 and clean.isalpha():
                            targets.add(clean)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error loading name_mapping.json: %s", exc)
    for key in HIDDEN_KB:                           # lecithin, surimi, tahini, marzipan ...
        if "_" not in key and len(key) >= 5:
            targets.add(key)
            tokens.add(key)
    for food in COMMON_FOODS:
        tokens.add(food)
        phrases.add(food)
        singles.add(food)
    tokens |= STRUCTURAL_VOCAB
    full = set(tokens) | ENGLISH_COMMON | LABEL_SAFE_WORDS
    return phrases, sorted(tokens), singles, full, targets


ALLERGEN_PHRASES, ALLERGEN_TOKENS, SINGLE_ALLERGENS, FULL_TOKEN_VOCAB, ALLERGEN_TARGETS = _load_allergen_vocab()
_TARGETS_BY_INITIAL: Dict[str, List[str]] = {}
for _t in sorted(ALLERGEN_TARGETS):
    _TARGETS_BY_INITIAL.setdefault(_t[0], []).append(_t)
# vocabulary used to judge whether a reading looks like a real label (3+ letters only)
READABLE_VOCAB = {w for w in (FULL_TOKEN_VOCAB | ALLERGEN_TARGETS) if len(w) >= 3}
GENERIC_FOOD_WORDS = {"natural", "water", "salt", "sea", "whole", "fiber", "protein", "original",
                      "powder", "extract", "seed", "seeds", "oil", "fat", "fats", "acid", "agent",
                      "agents", "flavor", "flavour", "flavors", "flavours", "spice", "spices", "free",
                      "line", "made", "may", "sugar", "palm", "cane", "dry", "pure", "rich"}


# -----------------------------------------------------------------------------
# OCR box
# -----------------------------------------------------------------------------
class OCRBox:
    def __init__(self, bbox: list, text: str, conf: float):
        self.raw_bbox = bbox
        self.text = str(text).strip()
        self.conf = float(conf)
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        self.x_min, self.x_max = float(min(xs)), float(max(xs))
        self.y_min, self.y_max = float(min(ys)), float(max(ys))
        self.w = max(1.0, self.x_max - self.x_min)
        self.h = max(1.0, self.y_max - self.y_min)
        self.cx = (self.x_min + self.x_max) / 2.0
        self.cy = (self.y_min + self.y_max) / 2.0
        self.area = self.w * self.h
        dx = bbox[1][0] - bbox[0][0]
        dy = bbox[1][1] - bbox[0][1]
        self.angle = math.atan2(dy, dx) if abs(dx) > 1e-3 else 0.0
        self.p = self.cy
        self.s = self.cx

    def __repr__(self) -> str:
        return f"OCRBox('{self.text}', conf={self.conf:.2f}, y=[{self.y_min:.0f}-{self.y_max:.0f}])"


def _rect_box(x0: float, y0: float, x1: float, y1: float, text: str, conf: float) -> OCRBox:
    return OCRBox([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, conf)


# -----------------------------------------------------------------------------
# Image loading, quality, geometry helpers
# -----------------------------------------------------------------------------
def load_pil(image_input: Any) -> Image.Image:
    """PIL / bytes / file-like (Streamlit UploadedFile) / numpy -> upright RGB PIL image."""
    if isinstance(image_input, Image.Image):
        pil = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(bytes(image_input)))
    elif hasattr(image_input, "getvalue"):                       # Streamlit UploadedFile
        pil = Image.open(io.BytesIO(image_input.getvalue()))
    elif hasattr(image_input, "read"):
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        pil = Image.open(io.BytesIO(image_input.read()))
    elif isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
    elif isinstance(image_input, (str, Path)):
        pil = Image.open(str(image_input))
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:  # noqa: BLE001
        pass
    if pil.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        rgba = pil.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        pil = bg
    elif pil.mode != "RGB":
        pil = pil.convert("RGB")
    long_side = max(pil.size)
    cap = CONFIG["max_input_long_side"]
    if long_side > cap:
        k = cap / long_side
        pil = pil.resize((max(1, int(pil.width * k)), max(1, int(pil.height * k))), Image.Resampling.LANCZOS)
    return pil


def assess_quality(rgb: np.ndarray) -> Dict[str, Any]:
    """Cheap checks that explain most 'the scanner does not work' cases. Returns metrics + warnings."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    k = 1000.0 / max(h, w)
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray
    blur_var = float(cv2.Laplacian(small, cv2.CV_64F).var())      # measured at <=1000 px so it is comparable
    mean = float(gray.mean())
    contrast = float(gray.std())
    clipped_hi = float((gray >= 250).mean())
    warnings: List[str] = []
    if max(h, w) < 900:
        warnings.append("The photo is low resolution. Small print will not be readable; move closer or use the phone's camera.")
    if blur_var < 40:
        warnings.append("The photo looks blurry. Hold the phone steady, tap to focus on the text, and retake.")
    if mean < 60:
        warnings.append("The photo is very dark. Add light (no flash glare) and retake.")
    if clipped_hi > 0.12:
        warnings.append("Large bright/glare areas detected. Tilt the pack a little to avoid reflections.")
    if contrast < 28:
        warnings.append("Very low contrast. Try better lighting or a flatter angle.")
    return {"width": w, "height": h, "blur_var": round(blur_var, 1), "brightness": round(mean, 1),
            "contrast": round(contrast, 1), "glare_fraction": round(clipped_hi, 3), "warnings": warnings}


def _resize_to_long(img: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    k = float(long_side) / max(h, w)
    if abs(k - 1.0) < 0.02:
        return img, 1.0
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=k, fy=k, interpolation=interp), k


def rotate_right_angle(img: np.ndarray, deg: int) -> np.ndarray:
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate by angle_deg (OpenCV convention: positive = counter-clockwise) without cropping corners."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _text_mask(gray: np.ndarray) -> np.ndarray:
    """Foreground = text pixels (the minority polarity), 0/1 uint8."""
    block = max(15, (min(gray.shape[:2]) // 40) | 1)
    dark = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
    light = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -12)
    return dark if dark.mean() <= light.mean() else light


def estimate_skew_deg(gray: np.ndarray) -> Tuple[float, float]:
    """
    Projection-profile skew search. Returns (correction_deg, gain).
    Apply with rotate_bound(img, correction_deg). gain = score(best)/score(0 deg).
    """
    small, _ = _resize_to_long(gray, 900)
    mask = _text_mask(small).astype(np.float32)
    h, w = mask.shape

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rot = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        prof = rot.sum(axis=1)
        return float(np.var(prof))

    span = CONFIG["skew_search_deg"]
    coarse = np.arange(-span, span + 0.01, 0.5)
    scores = [score(a) for a in coarse]
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - 0.5, best + 0.51, 0.1)
    fscores = [score(a) for a in fine]
    best = float(fine[int(np.argmax(fscores))])
    base = score(0.0)
    gain = (max(fscores) / base) if base > 0 else 1.0
    return best, gain


def clahe_gray(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)


def binarize_for_tesseract(rgb: np.ndarray) -> np.ndarray:
    gray = clahe_gray(rgb)
    block = max(25, (min(gray.shape[:2]) // 25) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 15)


# -----------------------------------------------------------------------------
# OCR engines
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# OCR engines & Multilingual Reader Management
# -----------------------------------------------------------------------------
SCRIPT_GROUPS: Dict[str, List[str]] = {
    "en": ["en"],
    "bn": ["bn", "en"],
    "hi": ["hi", "en"],
    "ur": ["ur", "en"],
    "ar": ["ar", "en"],
    "th": ["th", "en"],
    "zh-CN": ["ch_sim", "en"],
    "zh": ["ch_sim", "en"],
    "zh-TW": ["ch_tra", "en"],
    "ja": ["ja", "en"],
    "ko": ["ko", "en"],
    "fr": ["fr", "en"],
    "de": ["de", "en"],
    "es": ["es", "en"],
    "it": ["it", "en"],
    "pt": ["pt", "en"],
    "tr": ["tr", "en"],
    "ru": ["ru", "en"],
    "id": ["id", "en"],
    "ms": ["ms", "en"],
}

TESSERACT_LANG_MAP: Dict[str, str] = {
    "en": "eng",
    "bn": "ben",
    "hi": "hin",
    "ur": "urd",
    "ar": "ara",
    "th": "tha",
    "zh-CN": "chi_sim",
    "zh-TW": "chi_tra",
    "ja": "jpn",
    "ko": "kor",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "tr": "tur",
    "ru": "rus",
    "id": "ind",
    "ms": "msa",
}


class MultilingualReaderManager:
    """
    Step 4.1: LRU cache for EasyOCR readers across script groups.
    Keeps at most 2 or 3 readers in memory to prevent GPU / RAM exhaustion.
    """
    def __init__(self, max_readers: int = 3):
        self.max_readers = max_readers
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._gpu_available: Optional[bool] = None

    def _has_gpu(self) -> bool:
        if self._gpu_available is None:
            try:
                import torch
                self._gpu_available = bool(torch.cuda.is_available())
            except Exception:
                self._gpu_available = False
        return self._gpu_available

    def get_reader(self, lang: str = "en") -> Any:
        import easyocr
        norm_lang = lang if lang in SCRIPT_GROUPS else "en"
        key = norm_lang
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if len(self._cache) >= self.max_readers:
            evicted_lang, _ = self._cache.popitem(last=False)
            logger.info("Evicted EasyOCR reader '%s' from LRU cache", evicted_lang)

        lang_list = SCRIPT_GROUPS.get(norm_lang, ["en"])
        use_gpu = self._has_gpu()
        logger.info("Initializing EasyOCR reader for language '%s' (langs=%s, gpu=%s)...", norm_lang, lang_list, use_gpu)
        try:
            reader = easyocr.Reader(lang_list, gpu=use_gpu, verbose=False)
        except Exception as exc:
            logger.warning("Failed to initialize EasyOCR reader for %s: %s. Falling back to English.", lang_list, exc)
            if "en" in self._cache:
                return self._cache["en"]
            reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            key = "en"

        self._cache[key] = reader
        return reader


_READER_MANAGER: Optional[MultilingualReaderManager] = None


def get_reader_manager() -> MultilingualReaderManager:
    global _READER_MANAGER
    if _READER_MANAGER is None:
        _READER_MANAGER = MultilingualReaderManager(max_readers=3)
    return _READER_MANAGER


def get_ocr_reader(lang: str = "en"):
    """Singleton EasyOCR reader access through LRU cache."""
    return get_reader_manager().get_reader(lang=lang)


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def variants(self, rgb: np.ndarray, mode: str) -> List[Tuple[str, np.ndarray]]:
        out: List[Tuple[str, np.ndarray]] = []
        if mode in ("auto", "rgb"):
            out.append(("rgb", rgb))
        if mode in ("auto", "clahe"):
            out.append(("clahe", clahe_gray(rgb)))
        return out

    def read(self, img: np.ndarray) -> List[OCRBox]:
        reader = get_ocr_reader(self.lang)
        h, w = img.shape[:2]
        opts = dict(CONFIG.get("easyocr", {}))
        opts["paragraph"] = False
        opts["mag_ratio"] = 1.0
        opts["canvas_size"] = int(max(h, w))
        results = reader.readtext(img, **opts)
        boxes = []
        for bbox, text, conf in results:
            t = str(text).strip()
            if t and conf >= CONFIG["min_box_conf"]:
                boxes.append(OCRBox([[float(x), float(y)] for x, y in bbox], t, conf))
        return boxes


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def variants(self, rgb: np.ndarray, mode: str) -> List[Tuple[str, np.ndarray]]:
        out: List[Tuple[str, np.ndarray]] = []
        if mode in ("auto", "rgb"):
            out.append(("gray", cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
        if mode in ("auto", "clahe"):
            out.append(("binary", binarize_for_tesseract(rgb)))
        return out

    def read(self, img: np.ndarray) -> List[OCRBox]:
        import pytesseract
        tess_code = TESSERACT_LANG_MAP.get(self.lang, "eng")
        try:
            available = set(pytesseract.get_languages())
        except Exception:
            available = {"eng"}
        chosen_lang = tess_code if tess_code in available else "eng"

        data = pytesseract.image_to_data(
            img, lang=chosen_lang, config=f"--oem 1 --psm {CONFIG['tesseract_psm']}",
            output_type=pytesseract.Output.DICT)
        boxes = []
        for i, txt in enumerate(data["text"]):
            t = str(txt).strip()
            try:
                conf = float(data["conf"][i]) / 100.0
            except (TypeError, ValueError):
                continue
            if not t or conf < 0 or conf < CONFIG["min_box_conf"]:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append(_rect_box(x, y, x + w, y + h, t, conf))
        return boxes


def make_engines(engine: Optional[str] = None, lang: str = "en") -> List[Any]:
    choice = (engine or CONFIG["engine"]).lower()
    engines: List[Any] = []
    if choice in ("easyocr", "both"):
        try:
            import easyocr  # noqa: F401
            engines.append(EasyOCREngine(lang=lang))
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR not available: %s", exc)
    if choice in ("tesseract", "both") or (choice == "easyocr" and not engines):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            engines.append(TesseractEngine(lang=lang))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tesseract not available: %s", exc)
    if not engines and choice == "tesseract":
        try:
            import easyocr  # noqa: F401
            engines.append(EasyOCREngine(lang=lang))
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR not available: %s", exc)
    if not engines:
        raise RuntimeError("No OCR engine available. Install EasyOCR (pip install easyocr) and/or "
                           "Tesseract (sudo apt install tesseract-ocr; pip install pytesseract).")
    return engines


# -----------------------------------------------------------------------------
# Readability score (Step 4.3: script-aware)
# -----------------------------------------------------------------------------
def _clean_word(w: str) -> str:
    return re.sub(r"[^a-z]", "", _strip_accents(w.lower()))


def readability(boxes: List[OCRBox], expected_lang: Optional[str] = None) -> Dict[str, float]:
    """
    Step 4.3: Script-aware readability score.
    Works for any script:
      - mean recognition confidence
      - share of recognised characters belonging to dominant script
      - number of glossary hits (Route A glossary)
      - share of text lines looking like real food words
      - preserves identical English behavior when text is Latin/English.
    """
    if not boxes:
        return {"score": 0.0, "vocab_words": 0, "mean_conf": 0.0, "n": 0}

    combined_text = " ".join(b.text for b in boxes)
    script = detect_script(combined_text)
    mean_conf = sum(b.conf for b in boxes) / len(boxes)

    # 1. English / Latin evaluation (exact legacy logic preserved)
    vocab_score, vocab_words, plausible = 0.0, 0, 0.0
    for b in boxes:
        for raw in b.text.split():
            w = _clean_word(raw)
            if len(w) < 3:
                continue
            if w in READABLE_VOCAB or (w.endswith("s") and w[:-1] in READABLE_VOCAB):
                vocab_score += b.conf * len(w)
                vocab_words += 1
            elif len(w) >= 4 and w.isalpha():
                plausible += b.conf * len(w)

    legacy_score = vocab_score + 0.15 * plausible

    # If purely Latin script and has English tokens, return legacy score
    if script == "latin" and vocab_words >= 1:
        return {
            "score": legacy_score,
            "vocab_words": vocab_words,
            "mean_conf": mean_conf,
            "n": len(boxes),
            "script": script,
        }

    # 2. Multilingual Script-Aware Evaluation
    script_pat = SCRIPT_RANGES.get(script)
    if script_pat:
        matched_chars = len(script_pat.findall(combined_text))
        total_letters = max(1, sum(c.isalpha() for c in combined_text))
        script_share = matched_chars / total_letters
    else:
        script_share = 0.5

    glossary_hits = 0
    try:
        matcher = get_glossary_matcher()
        hits, _ = matcher.match_full_text(
            combined_text,
            lang=expected_lang if expected_lang and expected_lang != "auto" else None
        )
        glossary_hits = len(hits)
    except Exception:
        glossary_hits = 0

    foreign_tokens = 0
    for b in boxes:
        if script_pat and script_pat.search(b.text):
            foreign_tokens += 1
            vocab_score += b.conf * 5.0

    total_vocab_words = vocab_words + glossary_hits
    multilingual_score = (
        (vocab_score * 1.5)
        + (glossary_hits * 18.0)
        + (foreign_tokens * 4.0 * script_share)
        + (mean_conf * 12.0)
    )

    final_score = max(legacy_score, multilingual_score)
    return {
        "score": final_score,
        "vocab_words": total_vocab_words,
        "mean_conf": mean_conf,
        "n": len(boxes),
        "script": script,
    }


def _upright_ok(r: Dict[str, float]) -> bool:
    if r.get("script", "latin") == "latin":
        return (r["vocab_words"] >= 5 and r["score"] >= 18 and r["mean_conf"] >= 0.45) or (
            r["vocab_words"] >= 3 and r["score"] >= 24
        )
    return (r["vocab_words"] >= 2 or r["score"] >= 15) and r["mean_conf"] >= 0.35


def _median_box_height(boxes: List[OCRBox]) -> float:
    hs = [b.h for b in boxes if len(b.text) >= 3 and b.conf >= 0.3] or [b.h for b in boxes]
    return float(np.median(hs)) if hs else 0.0


# -----------------------------------------------------------------------------
# Full OCR pipeline: orientation -> skew -> adaptive zoom -> engines/variants
# -----------------------------------------------------------------------------
def run_ocr_pipeline(image_input: Any, engine: Optional[str] = None, label_lang: str = "auto") -> Dict[str, Any]:
    t0 = time.time()
    timings: Dict[str, float] = {}
    pil = load_pil(image_input)
    rgb0 = np.array(pil)
    quality = assess_quality(rgb0)
    engines = make_engines(engine, lang=label_lang)
    primary = engines[0]
    notes: List[str] = []

    long0 = max(rgb0.shape[:2])
    s1 = min(CONFIG["pass1_long_side"] / long0, 2.0)
    work = cv2.resize(rgb0, None, fx=s1, fy=s1, interpolation=cv2.INTER_AREA if s1 < 1 else cv2.INTER_CUBIC) \
        if abs(s1 - 1) > 0.02 else rgb0
    if work is rgb0:
        s1 = 1.0

    def pass1(img: np.ndarray) -> List[OCRBox]:
        name, arr = primary.variants(img, "rgb")[0]
        return primary.read(arr)

    # ---- orientation (0/90/180/270)
    t = time.time()
    boxes_by_angle: Dict[int, List[OCRBox]] = {0: pass1(work)}
    r0 = readability(boxes_by_angle[0], expected_lang=label_lang)
    angle = 0
    if not _upright_ok(r0):
        best_score = r0["score"]
        for deg in (90, 180, 270):
            boxes_by_angle[deg] = pass1(rotate_right_angle(work, deg))
            r = readability(boxes_by_angle[deg], expected_lang=label_lang)
            if r["score"] > best_score * 1.25 and r["score"] >= 6.0:
                best_score, angle = r["score"], deg
        if angle:
            notes.append(f"Rotated {angle} degrees")
    timings["orientation_s"] = round(time.time() - t, 2)
    rgb_o = rotate_right_angle(rgb0, angle)
    work_o = rotate_right_angle(work, angle)
    boxes1 = boxes_by_angle[angle]

    # ---- skew
    t = time.time()
    skew_applied = 0.0
    try:
        skew, gain = estimate_skew_deg(cv2.cvtColor(work_o, cv2.COLOR_RGB2GRAY))
        if abs(skew) >= CONFIG["skew_min_deg"] and gain >= CONFIG["skew_min_gain"]:
            skew_applied = skew
            rgb_o = rotate_bound(rgb_o, skew)
            work_o = rotate_bound(work_o, skew)
            boxes1 = pass1(work_o)
            notes.append(f"Straightened {skew:+.1f} degrees")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skew estimation failed: %s", exc)
    timings["skew_s"] = round(time.time() - t, 2)

    # ---- adaptive zoom: choose the scale from the measured text height
    h1 = _median_box_height(boxes1)                     # px in `work_o`
    final_img, reuse_pass1 = work_o, True
    if h1 > 0:
        h_orig = h1 / s1                                # px in the original photo
        long_o = max(rgb_o.shape[:2])
        s2 = CONFIG["target_box_height"] / max(h_orig, 1.0)
        s2 = min(s2, CONFIG["max_total_upscale"], CONFIG["max_pass2_long_side"] / long_o)
        s2 = max(s2, 0.5)
        if s2 > s1 * 1.15:
            interp = cv2.INTER_CUBIC if s2 > 1 else cv2.INTER_AREA
            final_img = cv2.resize(rgb_o, None, fx=s2, fy=s2, interpolation=interp)
            reuse_pass1 = False
            notes.append(f"Rescaled x{s2:.2f} (text lines were {h_orig:.0f}px tall)")
        if h_orig * max(s2, 1.0) < 16:
            quality["warnings"].append("Text is very small in this photo. Move closer so the ingredient list fills the frame.")

    # ---- engines x variants on the final image
    candidates: List[Dict[str, Any]] = []
    t = time.time()
    for eng in engines:
        best: Optional[Dict[str, Any]] = None
        for vname, arr in eng.variants(final_img, CONFIG["variant_mode"]):
            if eng is primary and vname == "rgb" and reuse_pass1 and CONFIG["variant_mode"] in ("auto", "rgb"):
                bx = boxes1
            else:
                bx = eng.read(arr)
            r = readability(bx, expected_lang=label_lang)
            cand = {"engine": eng.name, "variant": vname, "boxes": bx, "readability": r}
            if best is None or r["score"] > best["readability"]["score"]:
                best = cand
            candidates.append(cand)
            if CONFIG["variant_mode"] == "auto" and r["vocab_words"] >= 8 and r["mean_conf"] >= 0.55:
                break                                    # good enough, skip the extra variant
    timings["ocr_s"] = round(time.time() - t, 2)
    timings["total_s"] = round(time.time() - t0, 2)

    if not any(c["boxes"] for c in candidates):
        return {"boxes": [], "image": final_img, "angle": angle, "skew": skew_applied, "quality": quality,
                "candidates": candidates, "notes": notes, "timings": timings, "rgb_original": rgb0}
    best_c = max(candidates, key=lambda c: c["readability"]["score"])
    return {"boxes": best_c["boxes"], "image": final_img, "angle": angle, "skew": skew_applied,
            "quality": quality, "candidates": candidates, "best": best_c, "notes": notes,
            "timings": timings, "rgb_original": rgb0}


def extract_ocr_boxes(image_input: Any, engine: Optional[str] = None, label_lang: str = "auto") -> Tuple[List[OCRBox], np.ndarray, int]:
    """Backwards-compatible wrapper: (boxes, image, rotation)."""
    res = run_ocr_pipeline(image_input, engine, label_lang=label_lang)
    return res["boxes"], res["image"], res["angle"]


# -----------------------------------------------------------------------------
# Layout: multi-column detection and reading order
# -----------------------------------------------------------------------------
def split_columns(boxes: List[OCRBox], depth: int = 0) -> List[List[OCRBox]]:
    """
    Split into columns at the widest vertical gutter. The gutter threshold is based on the TEXT HEIGHT
    (not box width), so it also works when the OCR engine returns whole-line boxes.
    """
    if len(boxes) < 8 or depth >= 2:
        return [boxes]
    med_h = float(np.median([b.h for b in boxes]))
    thr = max(1.2 * med_h, 24.0)
    ordered = sorted(boxes, key=lambda b: b.x_min)
    right_edge = ordered[0].x_max
    best_gap, split_x = 0.0, None
    for i in range(1, len(ordered)):
        gap = ordered[i].x_min - right_edge
        if gap > best_gap and gap > thr and i >= 3 and (len(ordered) - i) >= 3:
            best_gap, split_x = gap, right_edge + gap / 2.0
        right_edge = max(right_edge, ordered[i].x_max)
    if split_x is None:
        return [boxes]
    left = [b for b in boxes if b.cx < split_x]
    right = [b for b in boxes if b.cx >= split_x]
    if len(left) < 3 or len(right) < 3:
        return [boxes]
    return split_columns(left, depth + 1) + split_columns(right, depth + 1)


def group_column_into_lines(boxes: List[OCRBox], is_rtl: bool = False) -> List[List[OCRBox]]:
    if not boxes:
        return []
    angles = [b.angle for b in boxes if abs(b.angle) < math.radians(45)]
    theta = float(np.median(angles)) if angles else 0.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    for b in boxes:
        b.p = -b.cx * sin_t + b.cy * cos_t
        b.s = b.cx * cos_t + b.cy * sin_t
    boxes_sorted = sorted(boxes, key=lambda b: b.p)
    median_h = float(np.median([b.h for b in boxes]))
    line_threshold = max(6.0, median_h * 0.48)
    lines: List[List[OCRBox]] = []
    refs: List[float] = []
    for b in boxes_sorted:
        placed = False
        for i, ref in enumerate(refs):
            if abs(b.p - ref) < line_threshold:
                lines[i].append(b)
                refs[i] = sum(item.p for item in lines[i]) / len(lines[i])
                placed = True
                break
        if not placed:
            lines.append([b])
            refs.append(b.p)
    for line in lines:
        if is_rtl:
            line.sort(key=lambda b: -b.s)
        else:
            line.sort(key=lambda b: b.s)
    order = sorted(range(len(lines)), key=lambda i: refs[i])
    return [lines[i] for i in order]


def group_boxes_into_lines(boxes: List[OCRBox], is_rtl: bool = False) -> List[List[OCRBox]]:
    all_lines: List[List[OCRBox]] = []
    for col in split_columns(boxes):
        all_lines.extend(group_column_into_lines(col, is_rtl=is_rtl))
    return all_lines


# -----------------------------------------------------------------------------
# Conservative typo healing
# -----------------------------------------------------------------------------
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _match_case(orig: str, new: str) -> str:
    if orig.isupper() and len(orig) > 1:
        return new.upper()
    if orig[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


_DIGIT_SUBS = {"0": "o", "1": "li", "5": "s", "8": "b", "3": "e", "4": "a", "6": "g"}
_PATTERN_SUBS = [("rn", "m"), ("vv", "w"), ("cl", "d")]
_CODE_LIKE = re.compile(                                      # E322, E150d, B12, D3, 30g, 2.5%, 120kcal
    r"^E\d{3,4}[a-z]?$|^[A-Z]\d{1,2}$|^\d+(?:[.,]\d+)?(?:mg|mcg|kcal|kj|kg|ml|cl|oz|lb|g|l|%)$", re.I)
_WORD_RE = re.compile(r"[\w'\u2019\-]*[^\W\d_][\w'\u2019\-]*")


def _is_known(w: str) -> bool:
    if w in FULL_TOKEN_VOCAB or w in ALLERGEN_TARGETS or w in SINGLE_ALLERGENS:
        return True
    if w.endswith("s") and (w[:-1] in FULL_TOKEN_VOCAB or w[:-1] in ALLERGEN_TARGETS):
        return True
    if w.endswith("es") and w[:-2] in FULL_TOKEN_VOCAB:
        return True
    return False


def de_glue_stuck_words(token: str) -> str:
    """'wheatflour' -> 'wheat flour'. Only splits into two known words of 3+ letters each."""
    t = token.lower()
    if len(t) < 7 or _is_known(t) or not t.isalpha():
        return token
    for i in range(3, len(t) - 2):
        a, b = t[:i], t[i:]
        if _is_known(a) and _is_known(b) and len(a) >= 3 and len(b) >= 3:
            return f"{token[:i]} {token[i:]}"
    return token


def _fuzzy_allergen(clean: str) -> Optional[str]:
    """Return an allergen word only when the match is very close. Never maps onto structural words."""
    if len(clean) < 5 or not clean.isalpha():
        return None
    first = clean[0]
    pool: List[str] = list(_TARGETS_BY_INITIAL.get(first, []))
    for alt in CONFUSABLE_FIRST.get(first, ""):
        pool.extend(_TARGETS_BY_INITIAL.get(alt, []))
    pool = [p for p in pool if abs(len(p) - len(clean)) <= 2]
    if not pool:
        return None
    res = process.extractOne(clean, pool, scorer=fuzz.ratio)
    if not res:
        return None
    cand, score = res[0], res[1]
    return cand if score >= (88 if len(clean) <= 6 else 84) else None


def heal_word(token: str) -> str:
    """Fix one word using unified spelling corrector in scanned mode."""
    res = spelling_corrector.correct_text(token, mode="scanned")
    if res.corrections:
        return res.corrections[0].corrected
    return token


def heal_line_tokens(line_text: str, log: Optional[List[Tuple[str, str]]] = None) -> str:
    """Heal line by line using the unified spelling corrector in scanned mode."""
    s = line_text.replace("[", "(").replace("]", ")").replace(";", ",")
    s = re.sub(r"([A-Za-z])\(", r"\1 (", s)
    s = re.sub(r"\b(ingredients?|contains?|advice|allergens?)_", r"\1:", s, flags=re.I)

    res = spelling_corrector.correct_text(s, mode="scanned")
    if log is not None:
        for c in res.corrections:
            log.append((c.original, c.corrected))

    out = re.sub(r"\s{2,}", " ", res.corrected_text).strip()
    return out


def merge_multiline_text(grouped_lines: List[List[OCRBox]], heal: bool = True,
                         log: Optional[List[Tuple[str, str]]] = None,
                         script: str = "latin") -> str:
    """Join lines (optionally healed), repairing words hyphenated across line breaks."""
    raw_lines = [" ".join(box.text for box in line) for line in grouped_lines]
    if not raw_lines:
        return ""

    # Step 4.6 & 8.4: Do NOT apply English spelling healing or hyphenation fix to non-Latin scripts!
    if script != "latin":
        processed = [normalize_multilingual_text(ln) for ln in raw_lines]
        return "\n".join(x for x in processed if x)

    processed = [heal_line_tokens(ln, log) if heal else re.sub(r"\s{2,}", " ", ln).strip() for ln in raw_lines]
    merged: List[str] = []
    i = 0
    while i < len(processed):
        cur = processed[i].strip()
        if not cur:
            i += 1
            continue
        if i + 1 < len(processed):
            nxt = processed[i + 1].strip()
            m = re.search(r"([A-Za-z]+)[\-~]$", cur)
            if m and nxt:
                stem = m.group(1).lower()
                words = nxt.split()
                mw = re.match(r"^([A-Za-z]+)(.*)$", words[0])
                if mw:
                    combined = stem + mw.group(1).lower()
                    if combined in FULL_TOKEN_VOCAB or combined in ALLERGEN_TARGETS or combined in SINGLE_ALLERGENS:
                        prefix = cur[:m.start()]
                        rest = " ".join(words[1:])
                        cur = (prefix + " " if prefix else "") + combined + mw.group(2) + (" " + rest if rest else "")
                        processed[i + 1] = ""
        merged.append(cur)
        i += 1
    return "\n".join(x for x in merged if x)


# -----------------------------------------------------------------------------
# Ingredient section extraction (never drops allergen warnings)
# -----------------------------------------------------------------------------
MULTILINGUAL_ING_HEADERS = (
    r"[il1|]?ngr[e\u00e9\u00e8a]d[il1|]{1,2}[eao]nts?|zutaten|inhaltsstoffe|ingredientes|ingredienti|"
    r"состав|ингредиенты|উপকরণ|উপাদান|উপাদানসমূহ|\u0938\u093e\u092e\u0917\u094d\u0930\u0940|अवयव|اجزاء|ترکیب|المكونات|المقادير|المحتويات|"
    r"配料|成分|配料表|原料|原材料名|原材料|ส่วนประกอบ|ส่วนผสม|komposisi|ramuan|bahan|içindekiler|malzemeler"
)
ING_HEADER_RE = re.compile(rf"(?:^|(?<=[\s\W]))(?:{MULTILINGUAL_ING_HEADERS})\s*[:.\-।॥]?", re.I)
CONFIRMED_HEADER_RE = re.compile(rf"(?:^|(?<=[\s\W]))(?:{MULTILINGUAL_ING_HEADERS})\s*[:।॥]", re.I)

MULTILINGUAL_WARN_TRIGGERS = (
    r"(?:may|might|could|can)\s+contain|traces?\s+of|"
    r"(?:produced|processed|made|manufactured|packed|packaged|handled)\s+(?:in|on)\s+[^.\n]*?(?:facility|factory|equipment|line)|"
    r"(?:allergy|allergen)\s+(?:advice|information|statement)|contains?\s*:|"
    r"হতে\s+পারে|উপস্থিত\s+থাকতে\s+পারে|অ্যালার্জি|ধারণ\s+করতে\s+পারে|"
    r"शामिल\s+हो\s+सकते\s+हैं|निशान|एलर्जी|"
    r"قد\s+يحتوي|أثر|حساسية|"
    r"peut\s+contenir|traces\s+de|allergènes|"
    r"kann\s+spuren|spuren\s+von|allergene|"
    r"puede\s+contener|trazas\s+de|alérgenos|"
    r"può\s+contenere|tracce\s+di|allergeni|"
    r"pode\s+conter|vestígios\s+de|alérgenos|"
    r"может\s+содержать|следы|аллергены|"
    r"可能含有|微量|过敏原|"
    r"を含む|アレルギー|コンタミネーション|"
    r"อาจมี|มี|สารก่อภูมิแพ้"
)
WARN_RE = re.compile(rf"(?:{MULTILINGUAL_WARN_TRIGGERS})[^.।॥\n]{{0,160}}", re.I)

_NUM = r"\d+(?:[.,]\d+)?"
NOISE_PATTERNS = [
    r"\bnutrition(?:al)?\s+(?:facts|information|info)\b",
    r"\btypical\s+values\b(?:\s+per\s+\d+\s*(?:g|ml))?",
    r"\bserving\s+size\s*:?\s*" + _NUM + r"\s*[a-z]{0,6}(?:\s*\(\s*" + _NUM + r"\s*(?:g|ml)\s*\))?",
    r"\bservings?\s+per\s+container\s*:?\s*" + _NUM,
    r"\bamount\s+per\s+serving\b",
    r"\b" + _NUM + r"\s*%\s*daily\s+value\b",
    r"\b(?:calories|total\s+fat|saturated\s+fat|trans\s+fat|cholesterol|sodium|total\s+carbohydrates?|"
    r"dietary\s+fib(?:er|re)|total\s+sugars|protein)\s*:?\s*" + _NUM + r"\s*(?:mg|mcg|g|kcal|kj|%)?",
    r"\bincludes\s+" + _NUM + r"\s*g\s+added\s+sugars\b",
    r"\b(?:distributed|manufactured|packed|imported)\s+(?:by|for)\b[^.,;]*",
    r"\bproduct\s+of\s+[A-Za-z]+(?:\s+[A-Za-z]+)?",
    r"\bstore\s+in\s+a\s+cool(?:\s+and)?\s+dry\s+place\b",
    r"\bkeep\s+refrigerated\b|\bshake\s+well\s+before\s+use\b|\bstore\s+away\s+from\s+heat\b",
    r"\b(?:best\s+before(?:\s+date)?|best\s+by|use\s+by|expiry(?:\s+date)?|exp|mfg|lot|batch)"
    r"(?:\s+(?:no|date))?\.?\s*[:.\-]?\s*(?:see\s+\w+(?:\s+\w+)?|[\w/\-.]*\d[\w/\-.]*)",
    r"\bplease\s+recycle\b|\bhow2recycle(?:\.info)?\b|\brecycle\s+me\b|\bwidely\s+recycled\b",
    r"\bquestions\s*\?|\bcall\s+1-\d{3}-\d{3}-\d{4}\b|\bvisit\s+us\s+at\s+\S+|\bwww\.\S+",
    r"\bnet\s+(?:wt|weight)\.?\s*:?\s*" + _NUM + r"\s*(?:oz|g|kg|lb|ml)\b(?:\s*\(\s*" + _NUM + r"\s*g\s*\))?",
]
NOISE_RES = [re.compile(p, re.I) for p in NOISE_PATTERNS]


def find_confirmed_header(text: str) -> Optional[re.Match]:
    """
    A confirmed header is an ingredients word that ends with a colon (or Indic danda) and is followed,
    within roughly the next 250 characters, by at least one separator.
    Only a confirmed header may be used to cut off text before it.
    """
    for m in CONFIRMED_HEADER_RE.finditer(text):
        trailer = text[m.end(): m.end() + 250]
        if any(sep in trailer for sep in (",", ";", "।", "॥", "、", "，", "・", "،", "؛", "\n")):
            return m
    return None


def _pick_header(text: str) -> Optional[re.Match]:
    """Return the first confirmed header, or None. No longer falls back to unconfirmed matches."""
    return find_confirmed_header(text)


def extract_clean_ingredient_section(full_text: str) -> str:
    """
    Returns: ingredient list + every allergen / trace / facility statement found anywhere in the text.
    Recall over precision: text is only removed when it is clearly not food (nutrition numbers, lot codes,
    addresses, recycling, urls). Nothing is cut off at the end.
    """
    unified = " ".join(l.strip() for l in str(full_text).splitlines() if l.strip())
    unified = re.sub(r"\s+", " ", unified)
    warnings: List[str] = []

    def grab(m: re.Match) -> str:
        warnings.append(m.group(0).strip(" ,;:"))
        return " . "

    body = WARN_RE.sub(grab, unified)
    header = _pick_header(body)
    if header:
        body = body[header.end():]
    for rx in NOISE_RES:
        body = rx.sub(" ", body)
    body = re.sub(r"\s*,(\s*,)+", ",", body)
    body = re.sub(r"\s+", " ", body).strip(" .,;:-")
    seen, uniq = set(), []
    for w in warnings:
        k = re.sub(r"\W+", " ", w.lower()).strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(w)
    parts = [p for p in [body] + uniq if p]
    return ". ".join(parts) + ("." if parts else "")


# -----------------------------------------------------------------------------
# Barcode
# -----------------------------------------------------------------------------
def ean_checksum_ok(code: str) -> bool:
    if not code.isdigit() or len(code) not in (8, 12, 13, 14):
        return False
    d = [int(c) for c in code]
    s = sum(x * (3 if i % 2 == 0 else 1) for i, x in enumerate(d[:-1][::-1]))
    return (10 - s % 10) % 10 == d[-1]


def _strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _strings(o)


def decode_barcodes(rgb: np.ndarray) -> List[str]:
    """Checksum-validated EAN/UPC codes (OpenCV >= 4.8 barcode module). [] if none or unsupported."""
    if not hasattr(cv2, "barcode"):
        return []
    try:
        det = cv2.barcode.BarcodeDetector()
    except Exception:  # noqa: BLE001
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    h, w = gray.shape[:2]
    found: List[str] = []
    for rot in (None, cv2.ROTATE_90_CLOCKWISE):
        g0 = gray if rot is None else cv2.rotate(gray, rot)
        for scale in (1.0, 0.5, 0.75, 1.5, 0.35, 2.0):
            s = min(scale, 2400.0 / max(h, w))
            g = g0 if abs(s - 1) < 1e-3 else cv2.resize(
                g0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            for blur in (False, True):
                im = cv2.GaussianBlur(g, (3, 3), 0) if blur else g
                try:
                    res = det.detectAndDecodeWithType(im)
                except Exception:  # noqa: BLE001
                    continue
                for code in _strings(res):
                    if ean_checksum_ok(code) and code not in found:
                        found.append(code)
                if found:
                    return found
    return found


# -----------------------------------------------------------------------------
# Front-of-pack
# -----------------------------------------------------------------------------
PROMOTIONAL_BADGES = {
    "new", "original", "classic", "family size", "party size", "value pack",
    "great taste", "improved recipe", "100%", "organic", "non gmo", "gluten free",
    "vegan", "delicious", "crispy", "crunchy", "fresh", "premium", "authentic",
    "natural", "homemade", "tradition", "quality",
}


def extract_front_pack_product_candidates(boxes: List[OCRBox], img_shape: Tuple[int, ...]) -> List[str]:
    """Brand/product title guesses ranked by text size x confidence (big text = brand/product name)."""
    if not boxes:
        return []
    img_h = img_shape[0]
    median_h = float(np.median([b.h for b in boxes])) or 1.0
    cands = []
    for b in boxes:
        t = b.text.lower().strip()
        if len(t) < 3 or re.match(r"^[\d\W]+$", t) or re.search(r"\b\d+\s*(?:g|oz|kg|ml|lb|l)\b", t):
            continue
        if t in PROMOTIONAL_BADGES:
            continue
        if any(w in t for w in ("nutrition", "calories", "servings", "ingredients", "best before")):
            continue
        score = (b.h / median_h) * b.conf
        if b.cy > 0.80 * img_h:
            score *= 0.5
        cands.append((b, score))
    cands.sort(key=lambda x: -x[1])
    if not cands:
        return []
    top = [c[0] for c in cands[:8]]
    lines = group_boxes_into_lines(top)
    proposals: List[str] = []

    def add(txt: str):
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) >= 3 and txt not in proposals:
            proposals.append(txt)

    if lines:
        biggest = max(lines, key=lambda ln: max(b.h for b in ln))
        add(" ".join(b.text for b in biggest))
    if len(lines) >= 2:
        add(" ".join(" ".join(b.text for b in ln) for ln in lines[:2]))
    for ln in lines:
        add(" ".join(b.text for b in ln))
    return proposals[:5]


_STOP = {"the", "and", "with", "for", "from", "de", "la", "le", "les", "des", "del"}


def _sig_tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]{3,}", _strip_accents(str(text).lower()))
    return [t for t in toks if t not in _STOP]


def _coverage(tokens: List[str], query_tokens: List[str]) -> Tuple[float, List[str]]:
    if not tokens or not query_tokens:
        return 0.0, []
    hit = [t for t in tokens if any(fuzz.ratio(t, q) >= 85 for q in query_tokens)]
    return len(hit) / len(tokens), hit


def verify_hits(query_texts: List[str], hits: List[dict]) -> List[dict]:
    """
    Keep only Open Food Facts hits whose name/brand really appear in the text read from the pack.
    score = 100 * (0.6 * share of product-name words read + 0.4 * share of brand words read),
    and at least one matching word must be distinctive (not a generic food word).
    """
    q_tokens = sorted({t for q in query_texts for t in _sig_tokens(q)})
    verified = []
    for h in hits:
        name_t, brand_t = _sig_tokens(h.get("name", "")), _sig_tokens(h.get("brand", ""))
        ncov, nhit = _coverage(name_t, q_tokens)
        if brand_t:
            bcov, bhit = _coverage(brand_t, q_tokens)
            score = 100.0 * (0.6 * ncov + 0.4 * bcov)
        else:
            bhit, score = [], 100.0 * ncov
        distinctive = [t for t in nhit + bhit if t not in GENERIC_FOOD_WORDS and t not in FULL_TOKEN_VOCAB]
        if score >= CONFIG["match_min_score"] and distinctive:
            verified.append({**h, "match_score": round(score, 1)})
    verified.sort(key=lambda r: (-r["match_score"], -int(r.get("scans", 0))))
    return verified


# -----------------------------------------------------------------------------
# Scan type
# -----------------------------------------------------------------------------
def classify_scan_type(full_text: str, boxes: List[OCRBox]) -> str:
    """'ingredient_list' only with real evidence; otherwise 'product_front'."""
    t = full_text.lower()
    if find_confirmed_header(full_text) or re.search(r"\bcontains?\s*:", t):
        return "ingredient_list"
    if re.search(r"\b(?:may|might|could)\s+contain\b|\btraces?\s+of\b", t) and t.count(",") >= 2:
        return "ingredient_list"
    try:
        matcher = get_glossary_matcher()
        hits, _ = matcher.match_full_text(full_text)
        if len(hits) >= 2 and any(sep in full_text for sep in (",", ";", "।", "॥", "、", "，", "\n")):
            return "ingredient_list"
    except Exception:
        pass
    words = re.findall(r"[a-z]{3,}", t)
    food_hits = sum(1 for w in words if (w in ALLERGEN_TARGETS or w in SINGLE_ALLERGENS) and w not in GENERIC_FOOD_WORDS)
    if t.count(",") >= 6 and food_hits >= 4:
        return "ingredient_list"
    return "product_front"


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def _save_debug(img: np.ndarray, boxes: List[OCRBox], tag: str) -> Optional[str]:
    out_dir = os.environ.get("ALLERSCAN_DEBUG_DIR")
    if not out_dir:
        return None
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    canvas = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for b in boxes:
        col = (0, 170, 0) if b.conf >= CONFIG["low_conf_word"] else (220, 0, 0)
        cv2.rectangle(canvas, (int(b.x_min), int(b.y_min)), (int(b.x_max), int(b.y_max)), col, 2)
    path = str(Path(out_dir) / f"{int(time.time() * 1000)}_{tag}.jpg")
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return path


def _allergen_set(text: str) -> set:
    try:
        allergens = set(parse_label(text).keys())
        matcher = get_glossary_matcher()
        hits, _ = matcher.match_full_text(text)
        for h in hits:
            if h.kind != "negated":
                allergens.add(h.canonical_key)
        return allergens
    except Exception:  # noqa: BLE001
        return set()


def _build_text(boxes: List[OCRBox], heal: bool, log: Optional[list] = None) -> Tuple[str, str]:
    sample = " ".join(b.text for b in boxes)
    script = detect_script(sample)
    is_rtl = (script == "arabic")
    lines = group_boxes_into_lines(boxes, is_rtl=is_rtl)
    merged = merge_multiline_text(lines, heal=heal, log=log, script=script)
    return merged, extract_clean_ingredient_section(merged)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def scan_photo(image_input: Any, mode: str = "auto", search_fn: Optional[Any] = None,
               engine: Optional[str] = None, label_lang: str = "auto") -> Dict[str, Any]:
    """
    mode: "auto" | "ingredient" | "product_front".  search_fn(query) -> list of
    {"name","brand","text","scans","code"} (app.search_products).
    label_lang: "auto" | "en" | "bn" | "hi" | "ar" | "ur" | ...
    """
    t_start = time.time()
    try:
        pipe = run_ocr_pipeline(image_input, engine, label_lang=label_lang)
    except RuntimeError as exc:
        return {"success": False, "error": str(exc), "mode": mode, "boxes": [], "raw_text": "", "cleaned_text": ""}
    quality = pipe["quality"]
    warnings: List[str] = list(quality["warnings"])
    rgb0 = pipe["rgb_original"]

    barcodes: List[str] = []
    if mode in ("auto", "product_front"):
        try:
            barcodes = decode_barcodes(rgb0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Barcode decoding failed: %s", exc)

    boxes: List[OCRBox] = pipe["boxes"]
    base: Dict[str, Any] = {
        "mode": mode, "quality": quality, "warnings": warnings, "barcodes": barcodes,
        "rotation_applied": pipe["angle"], "skew_applied": round(pipe["skew"], 2),
        "pipeline_notes": pipe["notes"], "timings": pipe["timings"],
        "engine_candidates": [{"engine": c["engine"], "variant": c["variant"],
                               "boxes": len(c["boxes"]), "score": round(c["readability"]["score"], 1),
                               "vocab_words": c["readability"]["vocab_words"]} for c in pipe["candidates"]],
        "matched_products": [], "product_candidates": [], "uncertain_allergens": [],
        "readings": [],
        "uncertain_trace_text": "", "low_confidence_words": [], "corrections": [],
    }

    # a barcode is the most reliable identity: look it up even if no text was read
    if barcodes and search_fn is not None and mode != "ingredient":
        for code in barcodes:
            try:
                hits = search_fn(code)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Barcode lookup failed for %s: %s", code, exc)
                hits = []
            if hits:
                base["matched_products"] = [{**h, "match_score": 100.0} for h in hits[:1]]
                base["best_query"] = code
                base["match_source"] = "barcode"
                break

    if not boxes:
        if base["matched_products"]:
            base.update(success=True, mode="product_front", raw_text="", cleaned_text="",
                        ingredients_text=base["matched_products"][0]["text"], boxes_count=0)
            return base
        base.update(success=False, boxes=[], raw_text="", cleaned_text="",
                    error="No text detected. " + (" ".join(warnings) if warnings else
                                                  "Try a sharper, well-lit photo with the label filling the frame."))
        if barcodes:
            base["error"] += f" (Barcode {barcodes[0]} was read but is not in Open Food Facts; try typing the ingredients.)"
        return base

    log: List[Tuple[str, str]] = []
    healed_text, cleaned = _build_text(boxes, heal=True, log=log)
    raw_text, cleaned_raw = _build_text(boxes, heal=False)

    detected = mode if mode in ("ingredient", "product_front") else classify_scan_type(healed_text, boxes)
    if detected == "ingredient":
        detected = "ingredient_list"

    # ---- allergens seen by ANY reading (engines/variants/raw/healed) but missing from the chosen text
    final_set = _allergen_set(cleaned)
    union = set(final_set) | _allergen_set(cleaned_raw)
    for c in pipe["candidates"]:
        if c is pipe.get("best") or not c["boxes"]:
            continue
        _, c_clean = _build_text(c["boxes"], heal=True)
        _, c_raw = _build_text(c["boxes"], heal=False)
        union |= _allergen_set(c_clean) | _allergen_set(c_raw)
    uncertain = sorted(union - final_set)
    base["uncertain_allergens"] = uncertain
    if uncertain:
        names = ", ".join(kb.pretty(a) for a in uncertain)
        base["uncertain_trace_text"] = (f"May contain traces of {', '.join(a.replace('_', ' ') for a in uncertain)} "
                                        f"(low-confidence photo reading, please verify).")
        warnings.append(f"A second reading of the photo also saw: {names}. Check the photo and the text before relying on it.")

    readings: List[Dict[str, Any]] = []
    for c in pipe["candidates"]:
        if not c["boxes"]:
            continue
        c_healed, c_clean = _build_text(c["boxes"], heal=True)
        readings.append({
            "label": f"{c['engine']}/{c['variant']}",
            "text": c_clean if c_clean else c_healed,
            "boxes_count": len(c["boxes"]),
            "score": round(c["readability"]["score"], 1),
            "is_chosen": bool(c is pipe.get("best")),
        })
    base["readings"] = readings

    base["corrections"] = [{"read": a, "healed_to": b} for a, b in log][:30]
    base["low_confidence_words"] = sorted(
        {b.text for b in boxes if b.conf < CONFIG["low_conf_word"] and len(b.text) >= 3})[:15]
    base["debug_image"] = _save_debug(pipe["image"], boxes, "boxes")
    base.update(success=True, raw_text=healed_text, raw_text_unhealed=raw_text, boxes_count=len(boxes))

    if detected == "ingredient_list":
        base.update(mode="ingredient_list", cleaned_text=cleaned, ingredients_text=cleaned)
    else:
        base["mode"] = "product_front"
        cands = extract_front_pack_product_candidates(boxes, pipe["image"].shape)
        base["product_candidates"] = cands
        base["cleaned_text"] = healed_text
        if not base["matched_products"] and search_fn and cands:
            for q in cands:
                try:
                    hits = search_fn(q)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Search error for '%s': %s", q, exc)
                    continue
                ok = verify_hits(cands, hits or [])
                if ok:
                    base["matched_products"], base["best_query"], base["match_source"] = ok, q, "name"
                    break
        if base["matched_products"]:
            base["ingredients_text"] = base["matched_products"][0]["text"]
        else:
            base["ingredients_text"] = cleaned or healed_text
            if search_fn and cands:
                warnings.append("No Open Food Facts product matched the brand/name that was read. "
                                "Scan the ingredient list on the back of the pack instead.")

    # Multilingual Analysis & Fusion (Step 8)
    mean_box_conf = (sum(b.conf for b in boxes) / len(boxes)) if boxes else 0.0
    text_to_analyze = cleaned if cleaned else healed_text
    multi_res: MultilingualAnalysisResult = analyze_multilingual_label(
        text=text_to_analyze,
        selected_language=label_lang if label_lang != "auto" else None,
        ocr_confidence=mean_box_conf,
    )
    base["multilingual"] = multi_res.to_dict()
    base["detected_language"] = multi_res.detected_language
    base["language_confidence"] = multi_res.language_confidence
    base["verification_state"] = multi_res.verification_state.value
    base["state_reason"] = multi_res.state_reason
    base["safe_verdict_allowed"] = multi_res.safe_verdict_allowed
    base["translated_text"] = multi_res.translated_text
    base["translation_engine"] = multi_res.translation_engine
    base["fused_allergens"] = {k: v.to_dict() for k, v in multi_res.allergens.items()}

    # Language mismatch check (Step 4.7)
    if label_lang != "auto" and multi_res.detected_language != "unknown" and multi_res.detected_language != label_lang:
        warnings.append(
            f"Selected label language is '{label_lang}', but detected script indicates '{multi_res.detected_language}'. "
            f"Please verify language selection."
        )

    # Verification state warning (Step 8.5)
    if multi_res.verification_state == VerificationState.COULD_NOT_VERIFY:
        warnings.append(f"Could not verify ingredients: {multi_res.state_reason}")

    if multi_res.detected_language != "en" and multi_res.translated_text:
        base["english_translated_text"] = multi_res.translated_text

    base["scan_seconds"] = round(time.time() - t_start, 2)
    return base


def merge_scan_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine several photos (e.g. both sides of a pack) into one ingredient text."""
    ok = [r for r in results if r.get("success")]
    if not ok:
        return {"success": False, "error": "; ".join(r.get("error", "") for r in results if r.get("error")) or "No text read.",
                "warnings": [w for r in results for w in r.get("warnings", [])]}
    texts = [r.get("ingredients_text", "") for r in ok if r.get("ingredients_text")]
    uncertain = sorted({a for r in ok for a in r.get("uncertain_allergens", [])} -
                       _allergen_set(" ".join(texts)))
    merged = dict(ok[0])
    merged["readings"] = []
    merged["ingredients_text"] = " ".join(t.strip() for t in texts)
    merged["cleaned_text"] = merged["ingredients_text"]
    merged["uncertain_allergens"] = uncertain
    merged["uncertain_trace_text"] = (
        f"May contain traces of {', '.join(a.replace('_', ' ') for a in uncertain)} "
        f"(low-confidence photo reading, please verify)." if uncertain else "")
    merged["warnings"] = list(dict.fromkeys(w for r in results for w in r.get("warnings", [])))
    merged["boxes_count"] = sum(r.get("boxes_count", 0) for r in ok)
    merged["photos_scanned"] = len(results)
    merged["photos_read"] = len(ok)
    merged["corrections"] = [c for r in ok for c in r.get("corrections", [])][:30]
    merged["low_confidence_words"] = sorted({w for r in ok for w in r.get("low_confidence_words", [])})[:15]
    return merged
