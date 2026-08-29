"""
Local RAG web app — loads the artifacts built by
v3_Local_RAG_Demo_top_k_with_window.ipynb and serves search + grounded answers.

    python v3_hello_rag.py     ->  http://localhost:5001

Needs: Ollama running, and rag_artifacts/ built by the notebook (Part 10).

The UI takes an "answer key" phrase alongside the question. Every retrieved
chunk is checked for it, so it is visible at a glance which rank holds the
answer, which ranks do not, and whether top-k was wide enough to send it on.
"""

import json
import re
from pathlib import Path

import faiss
import numpy as np
import ollama
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "rag_artifacts"

client = ollama.Client(host="http://localhost:11434", trust_env=False)

# ── Load artifacts once at startup ───────────────────────────────────────
config = json.loads((ARTIFACTS_DIR / "config.json").read_text())
EMBED_MODEL = config["EMBED_MODEL"]
GEN_MODEL = config["GEN_MODEL"]
WORDS_PER_CHUNK = config["WORDS_PER_CHUNK"]

with open(ARTIFACTS_DIR / "chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

index = faiss.read_index(str(ARTIFACTS_DIR / "faiss_index.bin"))
KEY_TO_IDX = {(c["source_path"], c["chunk_index"]): i for i, c in enumerate(chunks)}

print(f"{len(chunks)} chunks | {index.ntotal} vectors | embed={EMBED_MODEL} gen={GEN_MODEL}")


# ── Identical to the notebook ────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    v = np.asarray(
        client.embeddings(model=EMBED_MODEL, prompt=text)["embedding"], dtype="float32"
    )
    return v / (np.linalg.norm(v) + 1e-12)


def expand_with_neighbors_simple(I, chunks, neighbors=1, max_out=8):
    seen, contexts = set(), []
    for gi in I[0]:
        c = chunks[int(gi)]
        for delta in range(-neighbors, neighbors + 1):
            j = KEY_TO_IDX.get((c["source_path"], c["chunk_index"] + delta))
            if j is None or j in seen:
                continue
            seen.add(j)
            n = chunks[j]
            contexts.append({
                "title": n["title"],
                "source_path": n["source_path"],
                "chunk_index": n["chunk_index"],
                "text": n["text"],
                "approx_words": len(n["text"].split()),
                "is_hit": delta == 0,
            })
            if len(contexts) >= max_out:
                return contexts
    return contexts


# ── The answer key: same idea as `must_contain` in the notebook's gold set ─
def phrase_pattern(phrase: str):
    """Match `phrase` even if a line break landed between its words."""
    return re.compile(r"\s+".join(map(re.escape, phrase.split())), re.I)


def search_windowed(query, topk=3, neighbors=1, max_out=12):
    q_vec = embed_text(f"task: search result | query: {query}")
    D, I = index.search(q_vec.reshape(1, -1), max(topk, 5))
    contexts = expand_with_neighbors_simple(
        np.array([I[0][:topk]]), chunks, neighbors=neighbors, max_out=max_out
    )
    hits = [
        {
            "rank": r,
            "used": r <= topk,           # did this rank actually feed the context?
            "score": round(float(s), 3),
            "title": chunks[i]["title"],
            "source_path": chunks[i]["source_path"],
            "chunk_index": chunks[i]["chunk_index"],
            "text": chunks[i]["text"],
        }
        for r, (s, i) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1)
    ]
    return hits, contexts


def build_prompt(question, contexts):
    block = "\n\n".join(
        f"[{i}] ({c['title']}, chunk {c['chunk_index']})\n{c['text']}"
        for i, c in enumerate(contexts, start=1)
    )
    return (
        "Answer the question using ONLY the passages below. Cite the passage numbers "
        "you used, like [2].\n"
        "The passages are excerpts from novels, so the speaker of a line may be named "
        "in a neighbouring passage rather than the one containing the line — read them "
        "together before deciding who is speaking.\n"
        "If the passages genuinely do not answer the question, say so and state what "
        "they do show instead.\n\n"
        f"{block}\n\nQuestion: {question}\nAnswer:"
    )


# ── Routes ───────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "v3_index.html",
        embed_model=EMBED_MODEL,
        gen_model=GEN_MODEL,
        num_chunks=len(chunks),
    )


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("query") or "").strip()
    if not question:
        return jsonify({"error": "Empty query"}), 400

    topk = int(data.get("topk", 3))
    neighbors = int(data.get("neighbors", 1))
    generate = bool(data.get("generate", True))
    expect = (data.get("expect") or "").strip()
    pat = phrase_pattern(expect) if expect else None
    found = lambda t: bool(pat.search(t)) if pat else None

    try:
        hits, contexts = search_windowed(question, topk=topk, neighbors=neighbors)
        prompt = build_prompt(question, contexts)

        # Which context chunks are top-k hits in their own right? `is_hit` from
        # expand_with_neighbors_simple only marks the pass that emitted a chunk
        # first, so a later rank pulled in earlier as a neighbour loses its rank.
        rank_of = {(h["source_path"], h["chunk_index"]): h["rank"]
                   for h in hits if h["used"]}

        answer = None
        if generate:
            kw = {"think": False} if "qwen" in GEN_MODEL or "gpt-oss" in GEN_MODEL else {}
            resp = client.chat(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 250},
                **kw,
            )
            answer = resp["message"]["content"].strip()
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "query": question,
        "expect": expect,
        "answer": answer,
        "answer_has_key": found(answer) if answer else None,
        "prompt": prompt,
        "topk": topk,
        "hits": [dict(h, has_answer=found(h["text"])) for h in hits],
        "contexts": [
            {
                "n": i,
                "title": c["title"],
                "source": Path(c["source_path"]).name,
                "chunk_index": c["chunk_index"],
                "from_rank": rank_of.get((c["source_path"], c["chunk_index"])),
                "approx_words": c["approx_words"],
                "has_answer": found(c["text"]),
                "text": c["text"],
            }
            for i, c in enumerate(contexts, start=1)
        ],
    })


if __name__ == "__main__":
    # debug=False on purpose: the reloader would load the FAISS index twice.
    app.run(debug=False, port=5001)
