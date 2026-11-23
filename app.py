"""Multilingual Document OCR + Semantic Search. Run: python app.py -> http://localhost:5000"""
import json
import os
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import ocr
from search import SearchIndex, chunk_text, make_backend

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(BASE, "data"))
UPLOADS = os.path.join(DATA, "uploads")
INDEX_DIR = os.path.join(DATA, "index")
DOCS_JSON = os.path.join(DATA, "docs.json")
ALLOWED = ocr.IMAGE_EXT | {".pdf"}
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

docs = {}  # doc_id -> {id, name, pages:[{page,text,confidence,method}]}
if os.path.exists(DOCS_JSON):
    with open(DOCS_JSON, encoding="utf-8") as f:
        docs = {d["id"]: d for d in json.load(f)}

index = SearchIndex(make_backend())
if not index.load(INDEX_DIR) and docs:  # rebuild if index files are missing
    index.add_chunks([
        {"doc_id": d["id"], "doc": d["name"], "page": p["page"], "text": t}
        for d in docs.values() for p in d["pages"] for t in chunk_text(p["text"])
    ])


def persist():
    with open(DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(list(docs.values()), f, ensure_ascii=False, indent=1)
    index.save(INDEX_DIR)


def doc_summary(d):
    confs = [p["confidence"] for p in d["pages"] if p["text"]]
    return {"id": d["id"], "name": d["name"], "pages": len(d["pages"]),
            "chars": sum(len(p["text"]) for p in d["pages"]),
            "confidence": round(sum(confs) / len(confs), 1) if confs else 0.0}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/status")
def status():
    try:
        langs, missing = ocr.resolve_langs()
        err = None
    except ocr.OCRSetupError as e:
        langs, missing, err = None, [], str(e)
    return jsonify(search=index.info(), ocr_langs=langs, ocr_missing=missing,
                   ocr_error=err, docs=[doc_summary(d) for d in docs.values()])


@app.route("/api/upload", methods=["POST"])
def upload():
    enhance = request.form.get("enhance") == "true"
    try:
        langs, missing = ocr.resolve_langs(request.form.get("langs") or ocr.DEFAULT_LANGS)
    except ocr.OCRSetupError as e:
        return jsonify(error=str(e)), 500
    added, errors = [], []
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename or "")
        if os.path.splitext(name)[1].lower() not in ALLOWED:
            errors.append(f"{f.filename}: unsupported file type")
            continue
        doc_id = uuid.uuid4().hex[:8]
        path = os.path.join(UPLOADS, f"{doc_id}_{name}")
        f.save(path)
        try:
            pages = ocr.ocr_file(path, langs, enhance)
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        docs[doc_id] = {"id": doc_id, "name": name, "pages": pages}
        index.add_chunks([
            {"doc_id": doc_id, "doc": name, "page": p["page"], "text": t}
            for p in pages for t in chunk_text(p["text"])
        ])
        added.append(doc_summary(docs[doc_id]))
    persist()
    return jsonify(added=added, errors=errors, langs_used=langs, langs_missing=missing)


@app.route("/api/search")
def do_search():
    q = request.args.get("q", "")
    k = min(int(request.args.get("k", 5)), 20)
    return jsonify(query=q, mode=index.mode, results=index.search(q, k))


@app.route("/api/docs/<doc_id>")
def get_doc(doc_id):
    d = docs.get(doc_id)
    return (jsonify(d) if d else (jsonify(error="not found"), 404))


@app.route("/api/docs/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    if doc_id not in docs:
        return jsonify(error="not found"), 404
    del docs[doc_id]
    index.remove_doc(doc_id)
    persist()
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)
