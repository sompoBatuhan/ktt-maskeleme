from __future__ import annotations

import argparse
import base64
import csv
import difflib
import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KTT_DIR = HERE / "example_ktt"
OUT_DIR = HERE / "output" / "preprocessed11"
WORK_LONG = 2000
MAX_ATTEMPTS = 2
CHUNK = 10
IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

ANALYZE_PROMPT = """Bu bir 'Maddi Hasarli Trafik Kazasi Tespit Tutanagi' fotografi.
Sadece asagidaki JSON nesnesini dondur, baska hicbir sey yazma:
{"rotate_cw":0,"extra_content":false,"form_bbox":null,"blurry":false}

- rotate_cw: goruntuyu SAAT YONUNDE kac derece dondurursek yazilar duz/okunur
  olur? 0, 90, 180 veya 270 olabilir (form su an ters/yan duruyorsa 90/180/270).
- extra_content: kadrajda bu KTT formunun DISINDA baska bir belge, kagit, el,
  masa vb. goruniyor mu (true/false).
- form_bbox: extra_content true ise, SADECE bu KTT formuna ait alanin sinirlayici
  kutusu, 0-1000 normalize [sol,ust,sag,alt] tam sayi. extra_content false ise null.
- blurry: form ciddi derecede bulanik/okunaksiz mi (true/false)."""

VERIFY_PROMPT = """Bu duzeltilmis bir KTT (Maddi Hasarli Trafik Kazasi Tespit Tutanagi)
goruntusu. Sadece asagidaki JSON nesnesini dondur:
{"upright":true,"single_form":true,"legible":true}
- upright: form dik/okunur yonde mi (ters/yan degil).
- single_form: kadrajda sadece bu form var mi (baska belge/el yok).
- legible: yazilar secilebilir kalitede mi (asiri bulanik degil)."""


def setup_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


def ask_qwen(prompt: str, img_bytes: bytes, max_tokens: int) -> str:
    import httpx
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    proxy = os.getenv("PORTA_PROXY") or None
    if os.getenv("PORTA_USE_PROXY", "true").lower() == "false":
        proxy = None
    llm = ChatOpenAI(
        model="Qwen3.8-27B-FP8", base_url=os.environ["PORTA_BASE_URL"],
        api_key=os.getenv("PORTA_API_KEY", ""), http_client=httpx.Client(timeout=180.0, proxy=proxy),
        http_socket_options=(), temperature=0, max_tokens=max_tokens,
    )
    b64 = base64.b64encode(img_bytes).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])
    return llm.invoke([msg]).content


def parse_json_obj(raw: str) -> dict:
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("JSON nesnesi bulunamadi")
    return json.loads(s[a:b + 1])


def to_jpeg_bytes(im: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def load_fixed(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path))


def deskew_minarea(im: Image.Image) -> tuple[Image.Image, float]:
    gray = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return im, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return im, 0.0
    return im.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC), round(angle, 2)


def qwen_correct(path: Path, log: list[str]) -> np.ndarray:
    img = to_jpeg_bytes(load_fixed(path))
    attempt = 0
    while True:
        raw = ask_qwen(ANALYZE_PROMPT, img, 300)
        try:
            a = parse_json_obj(raw)
        except Exception as e:
            log.append(f"analyze parse HATASI: {e}")
            a = {"rotate_cw": 0, "extra_content": False, "form_bbox": None, "blurry": False}
        log.append(f"analyze[{attempt}] -> {a}")

        im = Image.open(io.BytesIO(img)).convert("RGB")
        W, H = im.size
        if a.get("extra_content") and a.get("form_bbox"):
            try:
                x1, y1, x2, y2 = [float(v) / 1000 for v in a["form_bbox"]]
                box = (max(0, round((x1 - 0.01) * W)), max(0, round((y1 - 0.01) * H)),
                       min(W, round((x2 + 0.01) * W)), min(H, round((y2 + 0.01) * H)))
                if box[2] - box[0] > 20 and box[3] - box[1] > 20:
                    im = im.crop(box)
                    log.append(f"kirpildi (fazla belge/el) -> {box}")
            except Exception as e:
                log.append(f"kirpma atlandi: {e}")
        rot = int(a.get("rotate_cw") or 0) % 360
        if rot:
            im = im.rotate(-rot, expand=True, fillcolor=(255, 255, 255))
            log.append(f"dondurildu: {rot} derece")
        im, ang = deskew_minarea(im)
        if ang:
            log.append(f"yamukluk duzeltildi: {ang} derece")
        img = to_jpeg_bytes(im)
        attempt += 1

        raw = ask_qwen(VERIFY_PROMPT, img, 200)
        try:
            v = parse_json_obj(raw)
            ok = bool(v.get("upright")) and bool(v.get("single_form")) and bool(v.get("legible"))
        except Exception as e:
            log.append(f"UYARI: verify parse HATASI, gorsel dogrulanmadan kabul edildi: {e}")
            v, ok = {}, True
        log.append(f"verify -> {v} (ok={ok})")
        if ok or attempt >= MAX_ATTEMPTS:
            break
    return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)


def make_clean(path: Path, out_dir: Path, force: bool) -> tuple[np.ndarray, list[str]]:
    log: list[str] = []
    f = out_dir / "clean" / f"{path.stem}_clean.jpg"
    if f.exists() and not force:
        log.append(f"duzeltilmis gorsel (onceki calistirma): clean/{f.name}")
        return cv2.imread(str(f)), log
    log.append("Qwen ile duzeltiliyor")
    img = qwen_correct(path, log)
    f.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return img, log


BLUR_KEYS = [
    "tckimlik", "kimlikno", "kimlikvergi", "vergino", "surucubelge", "belgeno", "alindigiyer", "adres",
    "telno", "ceptel", "eposta", "email", "sasino", "motorno", "markavemodel", "plaka", "kullanimsekli",
    "sigortalinin", "sirketininunvan", "sigortasirket", "acenteno", "policeno", "tramer", "baslangicbitis",
    "yesilkart", "ulke", "olke", "pasaport", "adisoyad", "adsoyad", "gorgu",
]
IMZA_KEYS = ["imza", "lmza", "mzasi", "surucusunun"]
PROTECT_KEYS = [
    "kirmiziisik", "kirmizisik", "tasitgiremez", "karsiyonden", "gecmeyasag", "kavsakta", "kavsakda",
    "yetkilimemur", "ayniistikamet", "aynistikamet", "arkadancarpmak", "sagadonus", "soladonus",
    "gerimanevra", "gecmesollama", "gecisoncelig", "parketme", "duraklama", "kurallarauygun", "hizdurum",
    "freniz", "tespitedildi", "uygunkutular",
]
NEUTRAL_KEYS = [
    "ilkdarbe", "aracinilk", "carpisma", "taslag", "gorusleri", "gorusler", "kazatarih", "kazayeri",
    "mahalle", "cadde", "sokak", "semt", "dikkat", "tutanag", "bilgileri", "gosteriniz",
]
NEUTRAL_OVERRIDE = ["imzalayiniz", "nushayida"]
KAZA_KEYS = {"kazatarih", "kazayeri", "mahalle", "cadde", "sokak", "semt"}
RULE_TAIL = {"freniz", "tespitedildi"}
RULE_INDEX = {
    "uygunkutular": -1, "kirmiziisik": 0, "kirmizisik": 0, "tasitgiremez": 1, "karsiyonden": 2,
    "gecmeyasag": 3, "kavsakta": 4, "kavsakda": 4, "yetkilimemur": 5, "ayniistikamet": 6,
    "aynistikamet": 6, "arkadancarpmak": 6, "sagadonus": 7, "soladonus": 8, "gerimanevra": 9,
    "gecmesollama": 10, "gecisoncelig": 11, "parketme": 12, "duraklama": 13, "kurallarauygun": 14,
    "hizdurum": 15, "freniz": 16, "tespitedildi": 16,
}
SHIFTED = {"arkadancarpmak"} | {k for k, v in RULE_INDEX.items() if v >= 7}


def fold(s: str) -> str:
    s = s.translate(str.maketrans("ıİşŞğĞüÜöÖçÇâÂ", "iIsSgGuUoOcCaA")).lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s.translate(str.maketrans("015", "ols"))


def find_key(t: str, keys: list[str]) -> tuple[str, float] | None:
    for k in keys:
        i = t.find(k) if len(k) > 5 else (0 if t.startswith(k) else -1)
        if i >= 0:
            return k, (i + len(k) / 2) / len(t)
    for k in keys:
        n = len(k)
        if n < 7 or len(t) < n - 2:
            continue
        for i in range(0, max(1, len(t) - n + 1)):
            if difflib.SequenceMatcher(None, t[i:i + n], k).ratio() >= 0.84:
                return k, min(1.0, (i + n / 2) / len(t))
    return None


def classify(text: str) -> tuple[str, str, float] | None:
    t = fold(text)
    if not t:
        return None
    if any(k in t for k in NEUTRAL_OVERRIDE):
        return "neutral", "override", 0.5
    if any(k in t for k in IMZA_KEYS) or t.startswith("l2ar"):
        return "blur", "imza", 0.5
    for typ, keys in (("protect", PROTECT_KEYS), ("blur", BLUR_KEYS), ("neutral", NEUTRAL_KEYS)):
        m = find_key(t, keys)
        if m:
            return typ, m[0], m[1]
    return None


def build_items(ocr: list) -> list[dict]:
    items = []
    for quad, text, _conf in ocr:
        c = classify(text)
        if not c:
            continue
        typ, key, pos = c
        q = np.asarray(quad, np.float32)
        tl, tr, br, bl = q
        top, bot = tl + (tr - tl) * pos, bl + (br - bl) * pos
        cx, cy = (top + bot) / 2
        items.append({
            "type": typ, "key": key, "text": text, "quad": q, "cx": float(cx), "cy": float(cy),
            "x1": float(q[:, 0].min()), "y1": float(q[:, 1].min()),
            "x2": float(q[:, 0].max()), "y2": float(q[:, 1].max()),
            "lh": float(max(8.0, np.linalg.norm(bl - tl))),
        })
    return items


def apply_m(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)


def rect_poly(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def rotation(W: int, H: int, angle: float) -> tuple[np.ndarray, int, int]:
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * sin + W * cos), int(H * cos + W * sin)
    M[0, 2] += nW / 2 - W / 2
    M[1, 2] += nH / 2 - H / 2
    return M, nW, nH


def estimate_angle(quads: list[np.ndarray]) -> float:
    angs, wts = [], []
    for q in quads:
        dx, dy = float(q[1][0] - q[0][0]), float(q[1][1] - q[0][1])
        if dx < 60:
            continue
        angs.append(math.degrees(math.atan2(dy, dx)))
        wts.append(dx)
    if len(angs) < 5:
        return 0.0
    order = np.argsort(angs)
    a, w = np.array(angs)[order], np.array(wts)[order]
    cum = np.cumsum(w)
    return float(a[np.searchsorted(cum, cum[-1] / 2)])


def ink_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
    return np.where(bh > 22, 255, 0).astype(np.uint8)


def detect_cells(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    H, W = ink.shape
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W // 25), 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, H // 40))))
    h = cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    v = cv2.dilate(v, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))
    grid = cv2.bitwise_or(h, v)
    n, _, stats, _ = cv2.connectedComponentsWithStats(255 - grid, connectivity=4)
    cells, min_area = [], 0.00015 * W * H
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < min_area or w < 20 or hh < 12:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + hh >= H - 1:
            continue
        if area > 0.2 * W * H or area / float(w * hh) < 0.55:
            continue
        cells.append((int(x), int(y), int(x + w), int(y + hh)))
    return cells


def vertical_lines(ink: np.ndarray, y1: float, y2: float, xa: float, xb: float) -> list[int]:
    y1, y2 = int(max(0, y1)), int(min(ink.shape[0], y2))
    xa, xb = int(max(0, xa)), int(min(ink.shape[1], xb))
    if y2 - y1 < 20 or xb - xa < 5:
        return []
    sub = cv2.dilate(ink[y1:y2, xa:xb], np.ones((1, 5), np.uint8))
    prof = (sub > 0).mean(axis=0)
    lines, i = [], 0
    while i < len(prof):
        if prof[i] >= 0.5:
            j = i
            while j < len(prof) and prof[j] >= 0.5:
                j += 1
            lines.append(xa + i + int(np.argmax(prof[i:j])))
            i = j
        else:
            i += 1
    return lines


def vertical_segments(ink: np.ndarray) -> list[dict]:
    H, W = ink.shape
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, H // 40))))
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(10, H // 60))))
    n, _, st, _ = cv2.connectedComponentsWithStats(v, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if h >= 0.10 * H and w <= 0.03 * W:
            segs.append({"x": x + w / 2, "x1": float(x), "x2": float(x + w), "y1": float(y), "y2": float(y + h)})
    return segs


def horizontal_lines(ink: np.ndarray, W: int) -> list[int]:
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (W // 5, 1)))
    rows = np.where((hl > 0).mean(axis=1) >= 0.25)[0]
    out: list[int] = []
    for y in rows:
        if not out or y - out[-1] > 4:
            out.append(int(y))
    return out


def first_group(lines: list[int]) -> list[int]:
    grp = [lines[0]]
    for x in lines[1:]:
        if abs(x - grp[-1]) <= 14:
            grp.append(x)
        else:
            break
    return grp


def lines_outward(ink: np.ndarray, top: float, bottom: float, start: float, dist: float, sign: int) -> list[int]:
    if sign < 0:
        return sorted(vertical_lines(ink, top, bottom, start - dist, start), reverse=True)
    return sorted(vertical_lines(ink, top, bottom, start, start + dist))


_ctx: dict = {}


def band_edge(ink: np.ndarray, top: float, bottom: float, txt_edge: float, W: int, sign: int) -> float | None:
    tokens, lh = _ctx.get("tokens", []), _ctx.get("lh", 20.0)
    letter = "A" if sign < 0 else "B"
    cand = [tk for tk in tokens if tk["kind"] == "letter" and tk["text"] == letter
            and top - 1.5 * lh <= tk["y1"] <= top + 2.5 * lh
            and 0 <= sign * (tk["cx"] - txt_edge) <= 0.12 * W]
    if cand:
        tk = min(cand, key=lambda t: abs(t["cx"] - txt_edge))
        edge_x = tk["x1"] if sign < 0 else tk["x2"]
        lines = lines_outward(ink, top, bottom, edge_x, 0.05 * W, sign)
        if lines:
            return first_group(lines)[-1] + sign * 3
        return edge_x + sign * 0.6 * lh

    mid = (top + bottom) / 2
    halves = ((top, mid), (mid, bottom))
    per_half = [lines_outward(ink, a, b, txt_edge, 0.10 * W, sign) for a, b in halves]
    tol = max(16.0, 0.015 * W)
    firsts = None
    for xu in per_half[0]:
        match = [xl for xl in per_half[1] if abs(xl - xu) <= tol]
        if match:
            firsts = (xu, min(match, key=lambda v: abs(v - xu)))
            break
    if firsts is None:
        return None
    edges = []
    for (a, b), f in zip(halves, firsts):
        grp = first_group(lines_outward(ink, a, b, f, 0.10 * W, sign))
        start = grp[-1] if grp and abs(grp[0] - f) <= 14 else f
        loose = lines_outward(ink, a, b, start + sign * 14, 0.06 * W, sign)
        edges.append(first_group(loose)[-1] if loose else start)
    _ctx["pairs"][sign] = ((top + mid) / 2, edges[0] + sign * 3, (mid + bottom) / 2, edges[1] + sign * 3)
    edge = min(edges) if sign < 0 else max(edges)
    return edge + sign * 3


def fit_rows(rules: list[dict], lh: float) -> tuple[float, float, int] | None:
    best = None
    for s in (0, 1):
        pts = [(RULE_INDEX[r["key"]] + (s if r["key"] in SHIFTED else 0), r["cy"])
               for r in rules if r["key"] in RULE_INDEX]
        slopes = [(y2 - y1) / (i2 - i1) for i1, y1 in pts for i2, y2 in pts if i2 - i1 >= 2]
        if not slopes:
            slopes = [(y2 - y1) / (i2 - i1) for i1, y1 in pts for i2, y2 in pts if i2 - i1 >= 1]
        if not slopes:
            continue
        p = float(np.median(slopes))
        if not (0.8 * lh <= p <= 3.5 * lh):
            continue
        a = float(np.median([y - p * i for i, y in pts]))
        res = float(np.median([abs(y - (a + p * i)) for i, y in pts]))
        if best is None or res < best[0]:
            best = (res, a, p, s)
    return None if best is None else (best[1], best[2], best[3])


def find_band(items: list[dict], cells: list, ink: np.ndarray, tokens: list[dict], W: int, H: int) -> dict | None:
    rules = [it for it in items if it["type"] == "protect"]
    _ctx["lh"] = float(np.median([r["lh"] for r in rules])) if rules else 20.0
    if len(rules) < 3:
        return None
    mx, my = float(np.median([r["cx"] for r in rules])), float(np.median([r["cy"] for r in rules]))
    rules = [r for r in rules if abs(r["cx"] - mx) < 0.22 * W and abs(r["cy"] - my) < 0.3 * H]
    if len(rules) < 3:
        return None
    lh = float(np.median([r["lh"] for r in rules]))
    body = [r for r in rules if r["key"] not in RULE_TAIL | {"uygunkutular", "hizdurum"}] or rules
    first_rule = min(r["y1"] for r in body)
    last_rule = max(r["y2"] for r in rules)
    txt_left, txt_right = min(r["x1"] for r in body), max(r["x2"] for r in body)

    fit = fit_rows(rules, lh)
    if fit:
        row0, pitch, shift = fit
    else:
        ys = sorted(r["cy"] for r in body)
        diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a > 0.5 * lh]
        pitch = float(np.median(diffs)) if len(body) >= 5 and diffs else 1.35 * lh
        row0, shift = None, 0

    header = [h for h in rules if h["key"] == "uygunkutular"]
    if header:
        top = min(h["y1"] for h in header) - 0.1 * lh
    else:
        tops = [row0 - 1.5 * pitch - 0.2 * lh] if row0 is not None else [first_rule - pitch - 0.2 * lh]
        for tk in tokens:
            if not (min(tops) - 2 * pitch <= tk["y1"] <= first_rule):
                continue
            if tk["kind"] == "letter" and txt_left - 0.12 * W <= tk["cx"] <= txt_right + 0.12 * W:
                tops.append(tk["y1"] - 0.3 * lh)
            elif tk["kind"] == "arac":
                tops.append(tk["y1"] - 0.8 * lh)
        top = max(min(tops), first_rule - 17 * pitch)
    gor = [it for it in items if it["key"] == "gorgu" and it["y2"] < first_rule]
    if gor:
        top = max(top, max(g["y1"] for g in gor) + 1.8 * lh)

    tail = [r for r in rules if r["key"] in RULE_TAIL]
    hiz = [r for r in rules if r["key"] == "hizdurum"]
    if tail:
        bottom = max(t["y2"] for t in tail) + 0.35 * lh
    elif hiz:
        bottom = max(h["y2"] for h in hiz) + pitch + 0.35 * lh
    elif row0 is not None:
        bottom = row0 + (16 + shift + 0.5) * pitch + 0.35 * lh
    else:
        bottom = last_rule + 2 * pitch + 0.3 * lh
    below = [it["y1"] for it in items if it["key"] in ("yesilkart", "ulke", "olke", "pasaport")
             and txt_left - 0.05 * W <= it["cx"] <= txt_right + 0.05 * W and it["y1"] > last_rule + 0.3 * pitch]
    below += [tk["y1"] for tk in tokens if tk["kind"] == "arac"
              and txt_left - 0.05 * W <= tk["cx"] <= txt_right + 0.05 * W and tk["y1"] > last_rule + 0.3 * pitch]
    if below:
        bottom = max(last_rule + 0.35 * lh, min(bottom, min(below) - 0.3 * lh))

    edges = {}
    for sign, txt_edge, letter in ((-1, txt_left, "A"), (1, txt_right, "B")):
        e = band_edge(ink, top, bottom, txt_edge, W, sign)
        how = "cizgi"
        if e is None:
            narrow = [c for c in cells if (c[2] - c[0]) < 0.045 * W and top <= (c[1] + c[3]) / 2 <= bottom
                      and (c[2] <= txt_edge and c[0] >= txt_edge - 0.10 * W if sign < 0
                           else c[0] >= txt_edge and c[2] <= txt_edge + 0.10 * W)]
            if len(narrow) >= 3:
                e = float(np.median([c[0] for c in narrow])) if sign < 0 else float(np.median([c[2] for c in narrow]))
                how = "hucre"
        if e is None:
            lt = [tk for tk in tokens if tk["kind"] == "letter" and tk["text"] == letter
                  and top - 2 * lh <= tk["y1"] <= first_rule]
            if lt:
                e = min(t["x1"] for t in lt) - 0.5 * lh if sign < 0 else max(t["x2"] for t in lt) + 0.5 * lh
                how = "harf"
        if e is None:
            e = txt_edge + sign * 0.045 * W
            how = "sabit"
        edges[sign] = (min(e, txt_left - 2) if sign < 0 else max(e, txt_right + 2), how)
    return {"left": float(edges[-1][0]), "right": float(edges[1][0]), "top": float(top), "bottom": float(bottom),
            "lh": lh, "pitch": pitch, "left_how": edges[-1][1], "right_how": edges[1][1], "rules": len(rules)}


def find_band_lines(items: list[dict], tokens: list[dict], ink: np.ndarray, W: int, H: int,
                    lh: float) -> dict | None:
    segs = vertical_segments(ink)
    left_c = [s for s in segs if 0.15 * W <= s["x"] <= 0.48 * W]
    right_c = [s for s in segs if 0.52 * W <= s["x"] <= 0.85 * W]
    best = None
    for a in left_c:
        for b in right_c:
            top, bot = max(a["y1"], b["y1"]), min(a["y2"], b["y2"])
            if bot - top < 0.15 * H or not (0.05 * H <= top <= 0.6 * H) or b["x"] - a["x"] < 0.2 * W:
                continue
            score = (bot - top) - 0.5 * (abs(a["y1"] - b["y1"]) + abs(a["y2"] - b["y2"]))
            if best is None or score > best[0]:
                best = (score, a, b, top, bot)
    if best is None:
        return None
    _, a, b, top, bottom = best
    span_top, span_bot = top, bottom

    def cluster(cands: list[dict], x0: float, sign: int) -> list[dict]:
        return [s for s in cands if 0 <= sign * (s["x"] - x0) <= 0.06 * W
                and min(s["y2"], span_bot) - max(s["y1"], span_top) >= 0.3 * (span_bot - span_top)]

    lc = list({id(s): s for s in cluster(left_c, a["x"], -1) + cluster(left_c, a["x"], 1) + [a]}.values())
    rc = list({id(s): s for s in cluster(right_c, b["x"], 1) + cluster(right_c, b["x"], -1) + [b]}.values())
    left, right = min(s["x1"] for s in lc), max(s["x2"] for s in rc)
    outer_l, outer_r = min(lc, key=lambda s: s["x"]), max(rc, key=lambda s: s["x"])

    top = max(top - 0.06 * H, min(s["y1"] for s in lc + rc))
    inside = [it for it in items if left <= it["cx"] <= right]
    header = [it for it in inside if it["key"] == "uygunkutular" and abs(it["y1"] - top) < 0.08 * H]
    if header:
        top = min(top, min(h["y1"] for h in header) - 0.1 * lh)

    inner_ends = [max(s["y2"] for s in grp) for grp in
                  ([s for s in lc if s is not outer_l], [s for s in rc if s is not outer_r]) if grp]
    tail = [it for it in inside if it["key"] in RULE_TAIL and top < it["y1"] < span_bot + 0.05 * H]
    hiz = [it for it in inside if it["key"] == "hizdurum" and top < it["y1"] < span_bot]
    pitch = (span_bot - top) / 18
    if inner_ends:
        inner = float(np.median(inner_ends))
        pitch = max(0.6 * lh, (inner - top) / 16)
    if tail:
        bottom = max(t["y2"] for t in tail) + 0.35 * lh
    elif hiz:
        bottom = max(h["y2"] for h in hiz) + pitch + 0.35 * lh
    elif inner_ends:
        bottom = inner + 2 * pitch + 0.35 * lh
    bottom = min(bottom, span_bot)
    below = [it["y1"] for it in inside if it["key"] in ("yesilkart", "ulke", "olke", "pasaport")
             and it["y1"] > top + 0.15 * H]
    if below:
        bottom = min(bottom, min(below) - 0.3 * lh)
    return {"left": left - 3, "right": right + 3, "top": float(top), "bottom": float(bottom), "lh": lh,
            "pitch": pitch, "left_how": "yapisal", "right_how": "yapisal", "rules": 0,
            "a_end": outer_l["y2"], "b_end": outer_r["y2"]}


def slant_inner_edge(poly: np.ndarray, band: dict) -> np.ndarray:
    pairs = _ctx.get("pairs", {})
    xs, ys = poly[:, 0], poly[:, 1]
    T, B = band["top"], band["bottom"]
    if abs(ys.min() - T) > 1.5:
        return poly
    for sign, edge_x in ((-1, band["left"] - 1), (1, band["right"] + 1)):
        if sign not in pairs:
            continue
        yu, xu, yl, xlo = pairs[sign]
        if abs(xu - xlo) < 5:
            continue
        inner = xs.max() if sign < 0 else xs.min()
        if abs(inner - edge_x) > 0.6:
            continue
        outer = xs.min() if sign < 0 else xs.max()
        y_end = ys.max()
        off = sign * 1
        inner_pts = [(xu + off, T), (xu + off, yu), (xlo + off, yl), (xlo + off, B), (edge_x, B), (edge_x, y_end)]
        if sign < 0:
            pts = [(outer, T)] + inner_pts + [(outer, y_end)]
        else:
            pts = [(outer, T), (outer, y_end)] + inner_pts[::-1]
        return np.array(pts, np.float32)
    return poly


def blur_regions(items: list[dict], tokens: list[dict], cells: list, ink: np.ndarray,
                W: int, H: int) -> tuple[list[np.ndarray], dict | None, list[str]]:
    notes: list[str] = []
    _ctx.clear()
    _ctx.update(tokens=tokens, lh=20.0, pairs={})
    xs1 = sorted(tk["x1"] for tk in tokens) or [0.05 * W]
    xs2 = sorted(tk["x2"] for tk in tokens) or [0.95 * W]
    o_left, o_right = xs1[int(0.03 * (len(xs1) - 1))], xs2[int(0.97 * (len(xs2) - 1))]
    wide = [c for c in cells if c[2] - c[0] > 0.25 * W]
    t_left = min([o_left] + [c[0] for c in wide])
    t_right = max([o_right] + [c[2] for c in wide])
    xl, xr = max(0.0, t_left - 0.05 * W), min(W - 1.0, t_right + 0.08 * W)
    lh_all = float(np.median([it["lh"] for it in items])) if items else 0.012 * H

    band = find_band(items, cells, ink, tokens, W, H)
    if band is None:
        band = find_band_lines(items, tokens, ink, W, H, lh_all)
        if band:
            notes.append("kural etiketleri okunamadi -> koruma bandi formun cizgilerinden bulundu")
            xl = min(xl, max(0.0, min(c[0] for c in wide) - 0.02 * W)) if wide else xl
    polys: list[np.ndarray] = []

    if band:
        L, R, T, B = band["left"], band["right"], band["top"], band["bottom"]

        def side_end(on_side, frame_end: float | None) -> float:
            ilk = [it for it in items if it["key"] in ("ilkdarbe", "aracinilk") and it["y1"] > B - 2 * band["lh"]]
            mine = [it["y2"] for it in ilk if on_side(it)]
            if mine:
                return min(mine)
            if ilk:
                return min(it["y2"] for it in ilk)
            if frame_end is not None:
                return frame_end
            bl = [it for it in items if it["type"] == "blur" and on_side(it) and T < it["y1"] < B + 0.25 * H
                  and it["key"] not in ("imza", "gorgu")]
            if bl:
                return max(i["y2"] for i in bl) + 0.9 * max(i["lh"] for i in bl)
            return B + 0.1 * H

        yA = min(max(side_end(lambda it: it["cx"] < L, band.get("a_end")), B), T + 0.6 * H)
        yB = min(max(side_end(lambda it: it["cx"] > R, band.get("b_end")), B), T + 0.6 * H)
        polys.append(rect_poly(xl, T, L - 1, yA))
        polys.append(rect_poly(R + 1, T, xr, yB))

        yk = [it for it in items if it["key"] in ("yesilkart", "ulke", "olke", "pasaport")
              and L - 0.05 * W <= it["cx"] <= R + 0.05 * W and B - band["lh"] < it["y1"] < B + 0.2 * H]
        if yk:
            y1 = B + 1
            y2 = max(it["y2"] for it in yk) + 0.6 * band["lh"]
            if not any(it["key"] == "pasaport" for it in yk):
                y2 += band["pitch"]
            y2 = min(y2, max(yA, yB) + 0.03 * H)
            if y2 > y1 + 10:
                polys.append(rect_poly(L, y1, R, y2))
                notes.append("yesil kart")
        elif band["left_how"] == "yapisal" and band.get("a_end") and band.get("b_end"):
            frame_end = min(band["a_end"], band["b_end"])
            if frame_end - B >= 0.055 * H and frame_end - band["pitch"] > B + 10:
                polys.append(rect_poly(L, B + 1, R, frame_end - band["pitch"]))
                notes.append("yesil kart (cizgilerden)")

        gor = [it for it in items if it["key"] == "gorgu" and it["y2"] < T]
        above = [it for it in items if it["type"] == "blur" and it["y2"] < T - 2 and it["key"] != "imza"]
        if gor or above:
            y1 = min([g["y1"] for g in gor] + [a["y1"] - 0.4 * a["lh"] for a in above])
            kaza = [it for it in items if it["key"] in KAZA_KEYS and it["y2"] < T]
            if kaza:
                y1 = max(y1, max(k["y2"] for k in kaza) + lh_all)
            dates = [tk["y2"] for tk in tokens if tk["y2"] < T - lh_all
                     and re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|^\d{1,2}[.:]\d{2}$", tk["text"])]
            if dates:
                y1 = max(y1, max(dates) + 0.3 * lh_all)
            if gor:
                y1 = min(y1, min(g["y1"] for g in gor) - 0.2 * lh_all)
            y2 = T - 1
            y1 = max(y1, y2 - 0.15 * H)
            if y2 > y1 + 10:
                polys.append(rect_poly(xl, y1, xr, y2))
                notes.append("gorgu")
        elif band["left_how"] == "yapisal":
            hl = sorted((y for y in horizontal_lines(ink, W) if T - 0.09 * H <= y < T - 4), reverse=True)
            y_top = T
            for y in hl:
                if y_top - y > 0.035 * H:
                    break
                y_top = y
            y1 = max(y_top, T - 0.07 * H)
            if T - y1 >= 0.02 * H:
                polys.append(rect_poly(xl, y1, xr, T - 1))
                notes.append("gorgu (tablo cizgilerinden)")
    else:
        mid = (t_left + t_right) / 2
        notes.append("UYARI: kural listesi bulunamadi, etiket bazli yedek yontem")
        for it in items:
            if it["type"] != "blur" or it["key"] in ("imza", "gorgu"):
                continue
            host = None
            for c in cells:
                ar = (c[2] - c[0]) * (c[3] - c[1])
                if c[0] <= it["cx"] <= c[2] and c[1] <= it["cy"] <= c[3] and ar < 0.03 * W * H \
                        and (c[3] - c[1]) < max(4 * it["lh"], 0.05 * H):
                    if host is None or ar < (host[2] - host[0]) * (host[3] - host[1]):
                        host = c
            if host:
                polys.append(rect_poly(*host))
            else:
                xt = mid - 2 if it["cx"] < mid else xr
                polys.append(rect_poly(it["x1"] - 3, it["y1"] - 0.55 * it["lh"], max(xt, it["x2"]),
                                       it["y2"] + 0.3 * it["lh"]))

    final = []
    for poly in polys:
        p = poly.copy()
        if band:
            ys, xs = p[:, 1], p[:, 0]
            if not (ys.max() < band["top"] or ys.min() > band["bottom"]):
                cx = xs.mean()
                if cx < band["left"]:
                    p[:, 0] = np.minimum(xs, band["left"] - 1)
                elif cx > band["right"]:
                    p[:, 0] = np.maximum(xs, band["right"] + 1)
                elif ys.max() > band["bottom"] + 10:
                    p[:, 1] = np.maximum(ys, band["bottom"] + 1)
                else:
                    continue
        if (p[:, 0].max() - p[:, 0].min()) > 4 and (p[:, 1].max() - p[:, 1].min()) > 4:
            final.append(p)
    if band:
        final = [slant_inner_edge(p, band) for p in final]
    return final, band, notes


def apply_blur(img: np.ndarray, polys: list[np.ndarray]) -> np.ndarray:
    out = img.copy()
    if not polys:
        return out
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [p.round().astype(np.int32) for p in polys], 255)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return out
    sigma = max(8.0, 0.02 * max(H, W))
    m = int(3 * sigma)
    x0, y0 = max(0, xs.min() - m), max(0, ys.min() - m)
    x1, y1 = min(W, xs.max() + m + 1), min(H, ys.max() + m + 1)
    roi = img[y0:y1, x0:x1]
    blurred = cv2.GaussianBlur(cv2.GaussianBlur(roi, (0, 0), sigma), (0, 0), sigma)
    sel = mask[y0:y1, x0:x1] > 0
    out[y0:y1, x0:x1][sel] = blurred[sel]
    return out


def find_polys(engine, clean: np.ndarray, log: list[str]) -> tuple[list[np.ndarray], list | None]:
    s = WORK_LONG / max(clean.shape[:2])
    img0 = cv2.resize(clean, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    H0, W0 = img0.shape[:2]

    res, _ = engine(img0)
    ocr = [[np.asarray(q, np.float32), t, float(c)] for q, t, c in (res or [])]
    label_quads = [it["quad"] for it in build_items(ocr)]
    ang = estimate_angle(label_quads if len(label_quads) >= 8 else [q for q, _, _ in ocr])

    M = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    W, H, img = W0, H0, img0
    if abs(ang) >= 0.4:
        for a in (ang, -ang):
            Mt, Wt, Ht = rotation(W0, H0, a)
            resid = estimate_angle([apply_m(q, Mt) for q in label_quads]) if len(label_quads) >= 5 else 0.0
            if abs(resid) < max(0.3, abs(ang) * 0.5):
                M, W, H = Mt, Wt, Ht
                img = cv2.warpAffine(img0, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
                log.append(f"egim duzeltildi: {ang:.2f} derece")
                break

    ocr_r = [[apply_m(q, M), t, c] for q, t, c in ocr]
    items = build_items(ocr_r)
    tokens = []
    for q, t, _c in ocr_r:
        f = fold(t)
        kind = "letter" if t.strip() in ("A", "B") else "arac" if f in ("araca", "aracb") else "text"
        tokens.append({"kind": kind, "text": t.strip(), "cx": float(q[:, 0].mean()),
                       "x1": float(q[:, 0].min()), "y1": float(q[:, 1].min()), "x2": float(q[:, 0].max()),
                       "y2": float(q[:, 1].max())})
    ink = ink_mask(img)
    cells = detect_cells(ink)
    polys, band, notes = blur_regions(items, tokens, cells, ink, W, H)
    log += notes
    if band:
        log.append(f"koruma bandi: sol={band['left_how']} sag={band['right_how']} kural_etiketi={band['rules']}")
    if not polys:
        log.append("UYARI: blur alani bulunamadi")

    Minv = cv2.invertAffineTransform(M)
    polys_clean = [apply_m(p, Minv) / s for p in polys]
    band_clean = None
    if band:
        bp = apply_m(rect_poly(band["left"], band["top"], band["right"], band["bottom"]), Minv) / s
        band_clean = bp.round(1).tolist()
    return polys_clean, band_clean


def safe_clean(path: Path, out_dir: Path, force: bool) -> tuple[np.ndarray | None, list[str], str]:
    try:
        img, log = make_clean(path, out_dir, force)
        if img is None:
            return None, log, "clean gorsel okunamadi"
        return img, log, ""
    except Exception as e:
        return None, [], f"{type(e).__name__}: {e}"


def process_chunk(engine, paths: list[Path], out_dir: Path, force: bool) -> list[dict]:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(paths)) as ex:
        cleans = list(ex.map(lambda p: safe_clean(p, out_dir, force), paths))
    per_clean = (time.time() - t0) / max(1, len(paths))

    rows = []
    for path, (clean, log, err) in zip(paths, cleans):
        print(f"\n=== {path.name} ===")
        rec: dict = {"file": path.name, "source": str(path)}
        t1 = time.time()
        try:
            if clean is None:
                raise RuntimeError(f"clean uretilemedi: {err}")
            polys, band_clean = find_polys(engine, clean, log)
            out = apply_blur(clean, polys)
            cv2.imwrite(str(out_dir / f"{path.stem}_blur.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_dir / f"{path.stem}_blur.json").write_text(json.dumps({
                "image": path.name, "source": str(path), "size": [clean.shape[1], clean.shape[0]],
                "blur_polygons": [p.round(1).tolist() for p in polys],
                "protected_band": band_clean, "log": log,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            for line in log:
                print(f"  - {line}")
            print(f"  sonuc: {len(polys)} blur alani -> {out_dir / (path.stem + '_blur.jpg')}")
            uyari = " | ".join(ln for ln in log if "UYARI" in ln)
            band = next((ln for ln in log if ln.startswith("koruma bandi:")), "YOK")
            rec.update(status="UYARI" if uyari else "OK", n_blur=len(polys),
                       band=band.replace("koruma bandi: ", ""),
                       qwen_duzeltme="evet" if any("Qwen ile duzeltiliyor" in ln for ln in log) else "",
                       uyari=uyari)
        except Exception as e:
            rec.update(status="ERROR", error=f"{type(e).__name__}: {e}")
            for line in log:
                print(f"  - {line}")
            print(f"  HATA: {rec['error']}")
        rec["secs"] = round(per_clean + time.time() - t1, 1)
        rows.append(rec)
    return rows


def resolve_inputs(names: list[str], use_all: bool, out_dir: Path) -> tuple[list[Path], list[str]]:
    found: list[Path] = []
    missing: list[str] = []
    if use_all:
        names = [str(KTT_DIR)] + list(names)
    for n in names:
        p = Path(n)
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
        elif p.is_file():
            found.append(p)
        elif (KTT_DIR / n).is_file():
            found.append(KTT_DIR / n)
        else:
            hits = sorted(f for f in KTT_DIR.iterdir() if f.name.startswith(n) and f.suffix.lower() in IMG_EXT)
            if hits:
                found += hits
            else:
                missing.append(n)
    seen, uniq = set(), []
    for f in found:
        r = f.resolve()
        if r in seen or out_dir in r.parents:
            continue
        seen.add(r)
        uniq.append(r)
    return uniq, missing


CSV_KEYS = ["file", "status", "n_blur", "band", "qwen_duzeltme", "uyari", "secs", "error", "source"]


def main() -> None:
    ap = argparse.ArgumentParser(description="KTT formunda kisisel veri alanlarini blur'la (imza haric).")
    ap.add_argument("inputs", nargs="*", help="gorsel dosyasi, klasor ya da example_ktt/ icinde ad (adin basi)")
    ap.add_argument("--all", action="store_true", help="example_ktt/ icindeki tum gorseller")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=CHUNK, help=f"parca buyuklugu (varsayilan {CHUNK})")
    ap.add_argument("--force", action="store_true", help="clean gorseli de Qwen ile yeniden uret")
    ap.add_argument("--out", default=str(OUT_DIR), help="cikti klasoru")
    args = ap.parse_args()
    if not args.inputs and not args.all:
        ap.error("en az bir gorsel/klasor ver ya da --all kullan")

    out = Path(args.out).resolve()
    images, missing = resolve_inputs(args.inputs, args.all, out)
    if args.limit:
        images = images[:args.limit]
    for img in images:
        if out == img.parent:
            raise SystemExit(f"cikti klasoru girdi gorsellerinin klasoruyle ayni olamaz: {out}")
    if out == KTT_DIR.resolve() or KTT_DIR.resolve() in out.parents:
        raise SystemExit("cikti klasoru example_ktt icinde olamaz")
    for n in missing:
        print(f"BULUNAMADI: {n}")
    if not images:
        raise SystemExit("islenecek gorsel yok")

    chunk = max(1, args.chunk)
    n_chunks = (len(images) + chunk - 1) // chunk
    print(f"{len(images)} gorsel, {n_chunks} parca ({chunk}'arli) -> {out}")
    out.mkdir(parents=True, exist_ok=True)
    setup_env()

    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    rows: list[dict] = [{"file": n, "status": "BULUNAMADI"} for n in missing]
    t_all = time.time()
    for k in range(n_chunks):
        part = images[k * chunk:(k + 1) * chunk]
        print(f"\n##### parca {k + 1}/{n_chunks} ({len(part)} gorsel) #####")
        rows += process_chunk(engine, part, out, args.force)
        with (out / "_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_KEYS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nbitti: {counts}  ({time.time() - t_all:.0f} sn)")
    print(f"cikti klasoru: {out}")


if __name__ == "__main__":
    main()

