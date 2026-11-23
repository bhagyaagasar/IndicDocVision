"""End-to-end checks. Run from the project folder:  python tests/test_pipeline.py
Uses SEARCH_BACKEND=tfidf so it needs no model download. Kannada is tested only
if samples/sample_kn.png exists (add your own page) and the 'kan' OCR language is installed."""
import difflib
import io
import os
import shutil
import sys
import tempfile

os.environ["SEARCH_BACKEND"] = "tfidf"
os.environ["DATA_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import ocr  # noqa: E402
from search import SearchIndex, chunk_text  # noqa: E402

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")
EN_TRUTH = ("The library is open Monday to Saturday from 9 AM to 5 PM. It is closed on Sundays and public holidays. "
            "Membership fee is Rs 500 per year.")
HI_TRUTH = ("पुस्तकालय सोमवार से शनिवार तक सुबह 9 बजे से शाम 5 बजे तक खुला रहता है। "
            "रविवार और सार्वजनिक अवकाश के दिन बंद रहता है। "
            "सदस्यता शुल्क 500 रुपये प्रति वर्ष है।")
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(("PASS" if ok else "FAIL"), "|", name, detail)


def sim(a, b):
    a, b = " ".join(a.split()), " ".join(b.split())
    return difflib.SequenceMatcher(None, a, b).ratio()


def best_window_sim(text, truth):
    """Similarity of `truth` against the best-matching same-length slice of OCR text."""
    t, n = " ".join(text.split()), len(truth)
    return max((sim(t[i:i + n], truth) for i in range(0, max(1, len(t) - n + 1), 10)), default=0)


langs, missing = ocr.resolve_langs("eng+hin")
check("tesseract has eng+hin", langs == "eng+hin", f"(using {langs})")

# 1. chunking
ch = chunk_text("First sentence. Second sentence.\n\nNew paragraph here। Another one।", size=30)
check("chunker keeps paragraphs apart", len(ch) >= 2 and not any("New paragraph" in c and "First" in c for c in ch))

# 2. OCR quality
en_pages = ocr.ocr_file(os.path.join(SAMPLES, "sample_en.png"), langs)
hi_pages = ocr.ocr_file(os.path.join(SAMPLES, "sample_hi.png"), langs)
s_en, s_hi = best_window_sim(en_pages[0]["text"], EN_TRUTH), best_window_sim(hi_pages[0]["text"], HI_TRUTH)
check("English OCR similarity > 0.9", s_en > 0.9, f"({s_en:.2f}, conf {en_pages[0]['confidence']})")
check("Hindi OCR similarity > 0.8", s_hi > 0.8, f"({s_hi:.2f}, conf {hi_pages[0]['confidence']})")

# 3. skewed + noisy scan, with enhancement
img = cv2.cvtColor(cv2.imread(os.path.join(SAMPLES, "sample_en.png")), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.5, 1.0)
skew = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
rng = np.random.default_rng(0)
noisy = np.clip(skew.astype(np.int16) + rng.normal(0, 18, skew.shape), 0, 255).astype(np.uint8)
text, conf = ocr.ocr_image(noisy, "eng", enhance=True)
s = best_window_sim(text, EN_TRUTH)
check("skewed+noisy scan with enhance > 0.8", s > 0.8, f"({s:.2f})")
_, ang = ocr.deskew(cv2.cvtColor(noisy, cv2.COLOR_RGB2GRAY))
check("deskew estimates the rotation", abs(abs(ang) - 3.5) <= 1.0, f"(estimated {ang:+.1f} deg for a 3.5 deg tilt)")

# 4. search (same-language)
idx = SearchIndex(backend=None)
for name, pages in (("en", en_pages), ("hi", hi_pages)):
    idx.add_chunks([{"doc_id": name, "doc": name, "page": 1, "text": t} for t in chunk_text(pages[0]["text"])])
r = idx.search("membership fee per year")
check("English query -> English doc", bool(r) and r[0]["doc_id"] == "en", f"(top: {r[0]['doc_id'] if r else None})")
r = idx.search("सदस्यता शुल्क कितना है")
check("Hindi query -> Hindi doc", bool(r) and r[0]["doc_id"] == "hi" and "शुल्क" in r[0]["text"],
      f"(top: {r[0]['doc_id'] if r else None})")
r = idx.search("fine for late return")
check("late-fee query finds the fine passage", bool(r) and "fine" in r[0]["text"].lower())
idx.remove_doc("en")
check("remove_doc works", all(c["doc_id"] != "en" for c in idx.chunks))

# 5. persistence
d = tempfile.mkdtemp()
idx.save(d)
idx2 = SearchIndex(backend=None)
check("index reload", idx2.load(d) and len(idx2.chunks) == len(idx.chunks))

# 6. web API
import app as A  # noqa: E402

c = A.app.test_client()
with open(os.path.join(SAMPLES, "sample_hi.png"), "rb") as f:
    r = c.post("/api/upload", data={"files": (io.BytesIO(f.read()), "sample_hi.png"), "langs": "eng+hin"},
               content_type="multipart/form-data").get_json()
check("API upload", len(r["added"]) == 1 and not r["errors"], str(r.get("errors")))
r = c.get("/api/search", query_string={"q": "सदस्यता शुल्क"}).get_json()
check("API search", bool(r["results"]) and "शुल्क" in r["results"][0]["text"])
bad = c.post("/api/upload", data={"files": (io.BytesIO(b"x"), "evil.exe")}, content_type="multipart/form-data").get_json()
check("API rejects unsupported file", bool(bad["errors"]) and not bad["added"])
doc_id = c.get("/api/status").get_json()["docs"][0]["id"]
check("API delete", c.delete(f"/api/docs/{doc_id}").status_code == 200 and not c.get("/api/status").get_json()["docs"])
check("home page", c.get("/").status_code == 200)

# 7. optional Kannada
kn = os.path.join(SAMPLES, "sample_kn.png")
if os.path.exists(kn) and "kan" in ocr.installed_langs():
    t = ocr.ocr_file(kn, "kan+eng")[0]
    check("Kannada OCR returned Kannada text", any("ಀ" <= ch <= "೿" for ch in t["text"]), f"(conf {t['confidence']})")
else:
    print("SKIP | Kannada OCR (add samples/sample_kn.png and install the 'kan' Tesseract language)")

shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)
print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
