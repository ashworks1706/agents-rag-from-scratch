"""The method library, as data: what each stage offers, a one-line blurb, and
a deep link to the spot in the deep-dive notebook that explains it.

The app uses this to build the pipeline from UI choices (no file editing) and
to render the Learn tab. Selectable stages are `splitting` and `searching`
(plus the reranker toggle); embedding and indexing are shown for reference,
since each search method wires in its own embedder and index.
"""

import importlib

# Deep-dive notebook, opened in Colab straight from GitHub.
NOTEBOOK_URL = (
    "https://colab.research.google.com/github/"
    "ashworks1706/agents-rag-from-scratch/blob/main/tutorial.ipynb"
)

# Section anchors in the notebook (cell metadata ids), used as a fallback when a
# method has no cell of its own.
SECTIONS = {
    "preprocessing": "5cc169c2",
    "storing": "00d238e7",
    "retrieval": "d8f8555b",
    "evaluation": "fdbce248",
}

# name -> (blurb, notebook cell id) linking to the cell that explains it.
SPLITTING = {
    "recursive": ("Split on paragraphs, then sentences, then words until chunks fit. The sensible default.", "9bb4bc33"),
    "character": ("Fixed-size character windows with overlap. Simple, ignores structure.", "5c3e786c"),
    "token_based": ("Chunk by token count (tiktoken) so chunks fit a model's context exactly.", "03d4aa70"),
    "sentence_nltk": ("Group whole sentences (NLTK) up to the size limit. No mid-sentence cuts.", "a22ff280"),
    "spacy_nlp": ("Sentence segmentation with spaCy's linguistic model.", "e5ee6c5a"),
    "markdown_header": ("Split by Markdown headers, keeping section structure.", "c837817a"),
    "html_header": ("Split by HTML headers, keeping section structure.", "3882b133"),
    "latex": ("Split LaTeX along sections and environments.", "4be7fee1"),
    "code_language": ("Split source code along language syntax boundaries.", "d75b44f7"),
}

SEARCHING = {
    "semantic_topk": ("Embed the query, return the k nearest chunks by cosine similarity. Dense retrieval.", SECTIONS["retrieval"]),
    "bm25_search": ("Classic keyword ranking (BM25). Strong lexical baseline, no embeddings.", "xaOoRJByincM"),
    "hybrid_search": ("Blend dense semantic and BM25 keyword scores.", "T05UrA4W-1pW"),
    "query_fusion": ("Generate query variations and merge their results.", "CqumoLmNAu2E"),
    "reciprocal_rank_fusion": ("Combine several rankings with reciprocal rank fusion (RRF).", "H7MlWpeyAwht"),
    "ensemble": ("Average scores from several retrievers.", "VmP2_izyB2iZ"),
    "router": ("Route each query to the retriever that suits it.", "sAKh2K7KB0tC"),
}

EMBEDDING = {
    "dense": ("sentence-transformers (all-MiniLM-L6-v2) turns text into normalized vectors.", "b006ffca"),
    "sparse_bm25": ("BM25 term-frequency sparse representation.", "eac78fbc"),
    "hybrid": ("Combine dense and sparse signals.", "5f7ea9f0"),
}

INDEXING = {
    "numpy_flat": ("Brute-force exact cosine over a NumPy matrix. Simple, fine to ~100k vectors.", "b673b1c0"),
    "faiss_index": ("FAISS vector index for scale.", "8848c209"),
    "hnsw_index": ("HNSW graph index. Fast approximate nearest neighbors.", "d594e1fa"),
    "lsh_index": ("Locality-sensitive hashing buckets.", "17bbf37d"),
    "chroma_index": ("Chroma persistent vector store.", "f0fe1616"),
}

RERANKING = {
    "cross_encoder": ("Re-score query-chunk pairs with a cross-encoder for precision.", "f369712e"),
}

# Plain-language definitions for the "?" help icons across the UI.
GLOSSARY = {
    "chunk": "A chunk is one slice of the document. The splitter cuts the text into "
             "chunks of about this many characters; retrieval and the answer work on chunks, not the whole doc.",
    "overlap": "How many characters each chunk repeats from the previous one. Overlap keeps "
               "a sentence that straddles a boundary from being lost between two chunks.",
    "top_k": "How many chunks to retrieve for each question. Higher k finds more but adds noise. "
             "Keep it the same across the room so scores compare fairly.",
    "reranker": "A second pass: a cross-encoder re-scores the retrieved chunks by reading each one "
                "together with the query, then keeps the best. Slower, usually more accurate.",
    "embedding": "A vector (list of numbers) that captures a chunk's meaning, so similar text lands "
                 "near it in space. Semantic search compares query and chunk embeddings.",
    "recall": "Recall@k: the fraction of questions whose answer appeared in your top-k chunks. The main score.",
    "mrr": "Mean Reciprocal Rank: 1 if the answer chunk was ranked first, 1/2 if second, and so on, averaged. "
           "Rewards ranking the right chunk high.",
    "answer_rate": "Answer@k: the fraction of questions the LLM answered with the gold phrase. Needs a Gemini key.",
}

# Rotating tips shown while the index builds.
TIPS = [
    "Chunks too big bury the answer in noise; too small and it loses context.",
    "Overlap keeps a sentence that straddles a boundary from being split across chunks.",
    "BM25 matches keywords; dense/semantic search matches meaning. Hybrid tries for both.",
    "Recall@k asks: did a chunk with the answer land in your top-k? It's the main score.",
    "MRR rewards ranking the right chunk first, not just somewhere in the top-k.",
    "A reranker reads each chunk together with the query. Slower, often more accurate.",
    "Swap methods in the sidebar and the app re-indexes automatically. No restart.",
    "Open the Learn tab to jump straight to the notebook section behind any method.",
    "Embeddings place similar text near each other; search finds the nearest neighbors.",
]


def coach_prompt(cfg):
    """A principle-based prompt users paste into their own LLM. It asks the LLM
    to teach the reasoning, not hand over the winning configuration.
    """
    reranker = "on" if cfg["use_reranker"] else "off"
    return f"""You are my coach for a hands-on RAG retrieval challenge. Teach me to reason about it. Do NOT hand me the answer or name the single best configuration.

The challenge: I have a fixed corpus and a set of questions. My pipeline retrieves the top-k text chunks for each question, and I'm scored on:
- Recall@k: did a chunk containing the answer make my top-k? (the main score)
- MRR: how high was the first correct chunk ranked?
- Answer@k (only if an LLM key is set): did the generated answer contain the gold phrase?

I can change these knobs, and only these:
- splitter: how the document is cut into chunks (currently: {cfg['splitter']}). Options include recursive, character, token, sentence, and header-aware splitters.
- chunk size / overlap (currently: {cfg['chunk_size']} / {cfg['overlap']}).
- search method (currently: {cfg['searcher']}). Options include BM25 keyword, dense/semantic, hybrid, query fusion, reciprocal rank fusion, ensemble, router.
- reranker on/off (currently: {reranker}).
- top_k is fixed for everyone, so I cannot just raise it.

Coach me like this:
1. Explain, in principle, what each knob changes about WHICH chunks get retrieved and HOW they get ranked, and the trade-offs (big vs small chunks, lexical vs semantic matching, when a reranker earns its cost).
2. Ask me diagnostic questions about where my current setup is likely losing points (outright misses vs correct-but-low-ranked).
3. Help me form and prioritize hypotheses to test myself, one change at a time, and how to read the score/rank chart after each run.

Never just state the winning combination. I want to understand it, not copy it. If I ask for "the best settings," push back and turn it into a question that makes me reason it out."""


# Ordered stages for the Learn tab. (selectable, folder, methods)
STAGES = [
    ("split", True, "splitting", SPLITTING),
    ("embed", False, "embedding", EMBEDDING),
    ("index", False, "indexing", INDEXING),
    ("search", True, "searching", SEARCHING),
    ("rerank", False, "reranking", RERANKING),
]


def colab_link(cell_id):
    return f"{NOTEBOOK_URL}#scrollTo={cell_id}"


def load_splitter(name):
    return importlib.import_module(f"splitting.{name}").split


def load_searcher(name):
    """The one class each searching/*.py file defines."""
    mod = importlib.import_module(f"searching.{name}")
    for value in vars(mod).values():
        if isinstance(value, type) and getattr(value, "__module__", None) == mod.__name__:
            return value
    raise ImportError(f"No search class found in searching.{name}")


def build_from_config(text, cfg, on_stage=None):
    """Split and index `text` per a UI config dict. Returns (chunks, searcher).

    cfg keys: splitter, searcher, chunk_size, overlap (top_k / use_reranker are
    applied at retrieval time, not here).
    """
    split = load_splitter(cfg["splitter"])
    chunks = split(text, cfg["chunk_size"], cfg["overlap"])
    if on_stage:
        on_stage("split", len(chunks))
    Search = load_searcher(cfg["searcher"])
    searcher = Search(chunks)
    if on_stage:
        on_stage("index", len(chunks))
    return chunks, searcher
