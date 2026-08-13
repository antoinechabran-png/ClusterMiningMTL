import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from pyvis.network import Network
from collections import Counter
import itertools
import re
import os
import json
import io
import math
import tempfile
import community as community_louvain  # python-louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semantic Explorer", layout="wide")

# ─── NLTK ───────────────────────────────────────────────────────────────────
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("omw-1.4",  quiet=True)
nltk.download("vader_lexicon", quiet=True)

# ─── Constants ──────────────────────────────────────────────────────────────
# Verbatim language support. English uses the existing, well-tested NLTK
# WordNet lemmatizer. The other five use spaCy's per-language models for
# real dictionary-form lemmatization (chosen over NLTK's Snowball stemmers
# for quality — at the cost of an extra dependency + a model download per
# language; see requirements.txt).
LANGUAGES = {
    "English":    {"code": "en", "nltk_stop": "english",    "spacy_model": None},
    "French":     {"code": "fr", "nltk_stop": "french",     "spacy_model": "fr_core_news_sm"},
    "Spanish":    {"code": "es", "nltk_stop": "spanish",    "spacy_model": "es_core_news_sm"},
    "Italian":    {"code": "it", "nltk_stop": "italian",    "spacy_model": "it_core_news_sm"},
    "German":     {"code": "de", "nltk_stop": "german",     "spacy_model": "de_core_news_sm"},
    "Portuguese": {"code": "pt", "nltk_stop": "portuguese", "spacy_model": "pt_core_news_sm"},
}

# Best-effort negation markers per language. NOTE: this is a simple "tag the
# word immediately following the marker" heuristic, same trick used for
# English — it works reasonably for languages where negation sits directly
# before the content word (Spanish, Portuguese) but is a rougher fit for
# French (discontinuous "ne...pas" wrapping the verb) and German (negation
# frequently lands at the end of the clause, not next to what it negates).
NEGATION_WORDS = {
    "en": ["not", "no", "don't", "can't", "won't", "never"],
    "fr": ["pas", "jamais", "aucun", "aucune", "non"],
    "es": ["no", "nunca", "jamás", "ningún", "ninguna", "tampoco"],
    "it": ["non", "mai", "nessuno", "nessuna", "niente"],
    "de": ["nicht", "kein", "keine", "nie", "niemals", "nichts"],
    "pt": ["não", "nunca", "jamais", "nenhum", "nenhuma", "tampouco"],
}

# Default exclusion words, translated per language (best-effort — editable
# by the user via the "Extra exclusion words" sidebar field either way).
DEFAULT_EXCLUSIONS_BY_LANG = {
    "English": [
        "product", "smell", "feel", "really", "just", "like", "little",
        "think", "lot", "make", "also", "bit", "quite", "something",
        "seem", "evoke", "find", "remind",
    ],
    "French": [
        "produit", "sentir", "ressentir", "vraiment", "juste", "comme",
        "petit", "penser", "beaucoup", "faire", "aussi", "peu", "assez",
        "quelque", "sembler", "évoquer", "trouver", "rappeler",
    ],
    "Spanish": [
        "producto", "oler", "sentir", "realmente", "solo", "como", "poco",
        "pensar", "mucho", "hacer", "también", "bastante", "algo",
        "parecer", "evocar", "encontrar", "recordar",
    ],
    "Italian": [
        "prodotto", "odore", "sentire", "davvero", "solo", "come", "poco",
        "pensare", "molto", "fare", "anche", "abbastanza", "qualcosa",
        "sembrare", "evocare", "trovare", "ricordare",
    ],
    "German": [
        "produkt", "riechen", "fühlen", "wirklich", "nur", "wie", "wenig",
        "denken", "viel", "machen", "auch", "bisschen", "ziemlich",
        "etwas", "scheinen", "hervorrufen", "finden", "erinnern",
    ],
    "Portuguese": [
        "produto", "cheirar", "sentir", "realmente", "apenas", "como",
        "pouco", "pensar", "muito", "fazer", "também", "bastante",
        "algo", "parecer", "evocar", "encontrar", "lembrar",
    ],
}

# Same palette as reference (5 clusters → extend if needed)
CLUSTER_COLORS = [
    "#0085AF",  # 1 – teal-blue
    "#E8A838",  # 2 – amber
    "#C62F4B",  # 3 – red
    "#6AAB6A",  # 4 – green
    "#8B6BB1",  # 5 – purple
    "#4BA8B0",  # 6
    "#E07B39",  # 7
    "#B85C8A",  # 8
    "#7B9E3E",  # 9
    "#D4724A",  # 10
]

# ─── Color helpers ────────────────────────────────────────────────────────────
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def darken_hex(hex_color, factor=0.4):
    """Auto-derive a matching border shade from any color, default or custom."""
    r, g, b = hex_to_rgb(hex_color)
    r, g, b = int(r * (1 - factor)), int(g * (1 - factor)), int(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"

def shade_rgb_str(hex_color, t):
    """t in [0,1] — 0 lightens toward white, 1 darkens toward black, 0.5 ≈ original.
    Used to give word-cloud words within a single focused cluster a readable
    frequency-driven variation while staying in that cluster's hue family."""
    r, g, b = hex_to_rgb(hex_color)
    if t < 0.5:
        f = (0.5 - t) * 2
        r = r + (255 - r) * f * 0.65
        g = g + (255 - g) * f * 0.65
        b = b + (255 - b) * f * 0.65
    else:
        f = (t - 0.5) * 2
        r = r * (1 - f * 0.55)
        g = g * (1 - f * 0.55)
        b = b * (1 - f * 0.55)
    return f"rgb({int(r)},{int(g)},{int(b)})"

def rgb_str(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    return f"rgb({r},{g},{b})"

def sentiment_to_color(score):
    """score in [-1,1] (VADER compound) → diverging red↔grey↔green hex."""
    score = max(-1.0, min(1.0, score if score is not None else 0.0))
    NEG, NEU, POS = (198, 47, 75), (222, 222, 222), (76, 175, 80)
    base = NEG if score < 0 else POS
    t = abs(score)
    r = base[0] * t + NEU[0] * (1 - t)
    g = base[1] * t + NEU[1] * (1 - t)
    b = base[2] * t + NEU[2] * (1 - t)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def normalize_sizes(values_dict, out_min=12, out_max=42):
    """Rescale an arbitrary metric (raw counts OR TF-IDF scores, whatever
    magnitude) onto a fixed pixel-size range, so node/bubble sizing looks
    sane regardless of which weighting metric is active."""
    if not values_dict:
        return {}
    vals = list(values_dict.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        mid = (out_min + out_max) / 2
        return {k: mid for k in values_dict}
    return {k: out_min + (v - lo) / (hi - lo) * (out_max - out_min) for k, v in values_dict.items()}

# Readable, dataviz-friendly fonts for the word cloud. Values are relative
# paths this app expects to find under a local "fonts/" folder — TTF files
# aren't bundled here (no network access to fetch them), so add the actual
# font files to your project for each entry you want to offer; anything
# missing falls back to the WordCloud default automatically.
FONT_OPTIONS = {
    "Default": None,
    "Inter":      "fonts/Inter-Regular.ttf",
    "Open Sans":  "fonts/OpenSans-Regular.ttf",
    "Roboto":     "fonts/Roboto-Regular.ttf",
    "Lato":       "fonts/Lato-Regular.ttf",
    "Montserrat": "fonts/Montserrat-Regular.ttf",
    "Nunito":     "fonts/Nunito-Regular.ttf",
}

# ─── NLP ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_lemmatizer():
    return WordNetLemmatizer()

@st.cache_resource
def load_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

@st.cache_resource
def load_spacy_model(model_name):
    """Loads a spaCy language model for lemmatization. Returns None (rather
    than raising) if the model isn't installed, so the app can fall back to
    basic tokenization instead of crashing — the caller is responsible for
    surfacing that as a warning."""
    if model_name is None:
        return None
    try:
        import spacy
        return spacy.load(model_name, disable=["parser", "ner"])
    except Exception:
        return None

@st.cache_resource
def load_stopwords(nltk_lang):
    try:
        return set(stopwords.words(nltk_lang))
    except Exception:
        return set()

# Unicode-aware word pattern — matches any run of Unicode letters (not just
# ASCII a-z). Without this, accented characters used throughout French,
# Spanish, Italian, German, and Portuguese (é, ñ, ü, ß, ã, ç, ...) would be
# silently stripped out or split words apart (e.g. "café" → "caf").
WORD_PATTERN = re.compile(r"\b[^\W\d_][^\W\d_]+\b", re.UNICODE)

def preprocess(text, lang_code, lemmatizer_en, spacy_nlp, custom_stops, stop_words):
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()

    neg_words = NEGATION_WORDS.get(lang_code, [])
    if neg_words:
        neg_pattern = r"\b(" + "|".join(re.escape(w) for w in neg_words) + r")\s+(\w+)"
        text = re.sub(neg_pattern, r"not_\2", text, flags=re.UNICODE)

    if lang_code == "en":
        tokens = WORD_PATTERN.findall(text)
        lemmas = [lemmatizer_en.lemmatize(t) for t in tokens]
    elif spacy_nlp is not None:
        # Feed spaCy the raw text directly rather than our own pre-tokenized
        # words — its tokenizer already understands each language's
        # contractions/clitics far better than a generic regex would.
        doc = spacy_nlp(text)
        lemmas = [tok.lemma_.lower() for tok in doc if tok.is_alpha]
    else:
        # No spaCy model installed for this language — fall back to plain
        # tokenization with no real lemmatization (plurals/verb forms won't
        # be reduced to a shared dictionary form).
        lemmas = WORD_PATTERN.findall(text)

    # Lemmatize/tokenize FIRST, then filter — the exclusion list should match
    # the word as it actually ends up counted/displayed, not the raw
    # inflected form it happened to take in the source text.
    return [
        lemma
        for lemma in lemmas
        if lemma not in stop_words and lemma not in custom_stops and len(lemma) > 2
    ]

def build_subcorpus_mask(texts, keywords):
    """Selects rows for an optional sub-corpus: True if the RAW verbatim text
    (before any tokenization/lemmatization) contains at least one of the
    given keywords/phrases as a whole word (case-insensitive). Deliberately
    NOT run on lemmatized tokens — see the sidebar help text for why:
    lemma behavior differs across languages (English is noun-only; the
    spaCy-backed languages are POS-aware), so raw-text matching is the only
    version that means the same thing regardless of language and matches
    what the user actually typed."""
    if not keywords:
        return [True] * len(texts)
    patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE | re.UNICODE) for kw in keywords]
    mask = []
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            mask.append(False)
            continue
        mask.append(any(p.search(t) for p in patterns))
    return mask

# ─── Spelling correction (optional) ────────────────────────────────────────────
@st.cache_resource
def load_spellchecker(lang_code):
    """pure-Python, dictionary-based, no compiled dependencies — deliberately
    avoided anything requiring a native build given past deployment pain.
    Returns None (rather than raising) if this language's dictionary isn't
    bundled, so the caller can degrade gracefully with a warning instead of
    crashing."""
    try:
        from spellchecker import SpellChecker
        # distance=1 (only single-character-edit fixes) rather than the
        # default 2 — more conservative, less likely to "correct" a
        # legitimate niche or brand-specific word into an unrelated one.
        return SpellChecker(language=lang_code, distance=1)
    except Exception:
        return None

def build_spelling_corrections(texts, spellchecker):
    """Corrects each UNIQUE unknown word once (not per occurrence — verbatim
    corpora repeat vocabulary heavily, so this avoids redundant work) and
    returns (correction_map, occurrences_corrected)."""
    word_counts = Counter()
    for t in texts:
        if isinstance(t, str):
            word_counts.update(WORD_PATTERN.findall(t.lower()))
    if not word_counts:
        return {}, 0
    unknown = spellchecker.unknown(word_counts.keys())
    correction_map = {}
    for w in unknown:
        corrected = spellchecker.correction(w)
        if corrected and corrected != w:
            correction_map[w] = corrected
    occurrences = sum(word_counts[w] for w in correction_map)
    return correction_map, occurrences

def apply_spelling_corrections(text, correction_map):
    if not isinstance(text, str) or not text.strip() or not correction_map:
        return text
    def repl(m):
        word = m.group(0)
        lw = word.lower()
        corrected = correction_map.get(lw)
        if not corrected:
            return word
        if word.isupper():
            return corrected.upper()
        if word[0].isupper():
            return corrected[:1].upper() + corrected[1:]
        return corrected
    return WORD_PATTERN.sub(repl, text)

def display_label(word):
    """Phrase tokens are stored internally as 'easy_apply' (a valid, unique
    graph/dict key); shown to the user as 'easy apply'."""
    return word.replace("_", " ")

# ─── Phrase (n-gram) detection ────────────────────────────────────────────────
def extract_top_bigrams(token_lists, min_freq=4, top_n=30):
    """Find recurring, meaningfully-associated adjacent word pairs (PMI-scored,
    within-row only — no bigrams manufactured across row boundaries)."""
    docs = [toks for toks in token_lists if len(toks) > 1]
    if not docs:
        return set()
    finder = BigramCollocationFinder.from_documents(docs)
    finder.apply_freq_filter(min_freq)
    if not finder.ngram_fd:
        return set()
    scored = finder.score_ngrams(BigramAssocMeasures.pmi)
    return set(bg for bg, _ in scored[:top_n])

def merge_bigrams(tokens, bigram_set):
    """Greedy, non-overlapping, left-to-right merge of adjacent tokens that
    form one of the detected phrases into a single 'word_word' token."""
    if not bigram_set:
        return tokens
    merged, i, n = [], 0, len(tokens)
    while i < n:
        if i < n - 1 and (tokens[i], tokens[i + 1]) in bigram_set:
            merged.append(tokens[i] + "_" + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged

# ─── Sentiment ────────────────────────────────────────────────────────────────
def compute_row_sentiment(texts, analyzer):
    if analyzer is None:
        return [None] * len(texts)
    scores = []
    for t in texts:
        if isinstance(t, str) and t.strip():
            try:
                scores.append(analyzer.polarity_scores(t)["compound"])
            except Exception:
                scores.append(None)
        else:
            scores.append(None)
    return scores

def compute_word_sentiment(token_lists, row_sentiments):
    sums, counts = Counter(), Counter()
    for tokens, sent in zip(token_lists, row_sentiments):
        if sent is None:
            continue
        for w in set(tokens):
            sums[w] += sent
            counts[w] += 1
    return {w: sums[w] / counts[w] for w in sums}

def cluster_avg_sentiment(members, word_freq, word_sent):
    num = sum(word_sent.get(w, 0.0) * word_freq.get(w, 0) for w in members if w in word_sent)
    den = sum(word_freq.get(w, 0) for w in members if w in word_sent)
    return (num / den) if den else None

# ─── TF-IDF weighting ──────────────────────────────────────────────────────────
def compute_tfidf_scores(token_lists):
    docs = [" ".join(toks) for toks in token_lists if toks]
    if not docs:
        return {}
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(), preprocessor=lambda x: x,
        token_pattern=None, lowercase=False,
    )
    matrix = vectorizer.fit_transform(docs)
    sums = matrix.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    return dict(zip(vocab, sums))

# ─── Clustering ────────────────────────────────────────────────────────────────
def cluster_to_target(G, target_n, seed=42):
    """Louvain community detection doesn't take a 'number of clusters' input —
    it optimizes modularity and lands wherever the graph's structure implies.
    Random seed barely moves that (it mostly just breaks ties), which is why
    searching over seeds alone was unreliable. The actual lever for cluster
    granularity is Louvain's `resolution` parameter — higher values favor
    more, smaller communities; lower values favor fewer, larger ones — so we
    search over that instead.

    Hard floor: modularity optimization can never merge two fully
    disconnected pieces of the graph into one community — doing so always
    makes modularity worse, so no resolution value will ever do it. That
    means the number of connected components in G is a hard lower bound on
    how few clusters are achievable, independent of resolution.

    Returns (partition, achieved_diff, n_components) — achieved_diff is 0
    only if the target was hit exactly; n_components is the hard floor
    explained above."""
    n_components = nx.number_connected_components(G)
    resolutions = [
        0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0,
        7.0, 10.0, 15.0, 20.0,
    ]
    best_p, best_diff = None, 10 ** 9
    for res in resolutions:
        p = community_louvain.best_partition(G, resolution=res, random_state=seed)
        diff = abs(len(set(p.values())) - target_n)
        if diff < best_diff:
            best_diff, best_p = diff, p
        if diff == 0:
            break
    return best_p, best_diff, n_components

# ─── Network builder ─────────────────────────────────────────────────────────
def build_html(G, partition, word_freq, color_map, size_map=None, word_sent=None, filename="semantic_map"):
    cluster_ids = sorted(set(partition.values()))
    size_map = size_map or word_freq
    word_sent = word_sent or {}
    has_sentiment = bool(word_sent)

    pixel_sizes = normalize_sizes(
        {n: size_map.get(n, word_freq.get(n, 1)) for n in G.nodes()}, 12, 42
    )

    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333333")

    # Ground-truth per-node styling, computed once in Python. This — and NOT
    # anything read back out of the live vis.js DataSet at click-time — is
    # what the JS below uses to restore colors. Reading "originals" out of the
    # rendered network after it may already have been mutated by a previous
    # fade/highlight is what caused clusters after the first to render wrong
    # and "All" to come back grey.
    node_meta = {}
    node_sent = {}

    for node in G.nodes():
        cluster = partition[node]
        freq    = G.nodes[node].get("size", 10)
        x       = G.nodes[node].get("x", 0)
        y       = G.nodes[node].get("y", 0)
        color = {
            "background": color_map[cluster],
            "border":     darken_hex(color_map[cluster]),
            "highlight":  {"background": "#FF8000", "border": "#CC5500"},
        }
        font = {"size": 13, "color": "#ffffff", "face": "Arial",
                "strokeWidth": 2, "strokeColor": "rgba(0,0,0,0.3)"}
        node_meta[node] = {"color": color, "font": font, "group": str(cluster)}

        if has_sentiment:
            sc = sentiment_to_color(word_sent.get(node, 0.0))
            node_sent[node] = {
                "color": {
                    "background": sc,
                    "border": darken_hex(sc),
                    "highlight": {"background": "#FF8000", "border": "#CC5500"},
                },
                "font": {"size": 13, "color": "#222222", "face": "Arial",
                         "strokeWidth": 2, "strokeColor": "rgba(255,255,255,0.6)"},
                "group": str(cluster),
            }

        title = f"<b>{display_label(node)}</b><br>Occurrences: {freq}<br>Cluster: {cluster + 1}"
        if has_sentiment and node in word_sent:
            title += f"<br>Sentiment: {word_sent[node]:+.2f}"

        net.add_node(
            node,
            label=display_label(node),
            title=title,
            color=color,
            size=pixel_sizes.get(node, 20),
            shape="box",
            group=str(cluster),
            x=x, y=y,
            physics=False,
            font=font,
            borderWidth=2,
            shadow={"enabled": True, "color": "rgba(0,0,0,0.15)", "size": 6, "x": 2, "y": 2},
        )

    for u, v, data in G.edges(data=True):
        net.add_edge(
            u, v,
            value=data.get("weight", 1),
            color={"color": "#c8d8e8", "highlight": "#FF8000", "opacity": 0.7},
            smooth=False,
        )

    # Physics off, hover on, same zoom speed as reference
    net.set_options("""{
      "physics": {"enabled": false},
      "interaction": {"hover": true, "zoomSpeed": 1},
      "edges": {"smooth": false}
    }""")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()

    # ── Cluster legend pills ──────────────────────────────────────────────────
    legend_pills = ""
    for i, c in enumerate(cluster_ids):
        members = sorted(
            [w for w, cl in partition.items() if cl == c],
            key=lambda w: -word_freq.get(w, 0),
        )
        top = display_label(members[0]).upper() if members else f"C{i+1}"
        col = color_map[c]
        tooltip_words = ", ".join(display_label(w) for w in members[:6])
        legend_pills += (
            f'<div onclick="filterCluster({c})" '
            f'style="background:{col};color:#fff;padding:6px 14px;border-radius:20px;'
            f'cursor:pointer;font-size:12px;font-weight:bold;white-space:nowrap;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.18);user-select:none;" '
            f'title="{tooltip_words}">'
            f'● C{i+1} – {top}'
            f'</div>\n'
        )

    sent_toggle_html = ""
    if has_sentiment:
        sent_toggle_html = (
            '<span style="width:1px;height:20px;background:#ddd;margin:0 2px;"></span>'
            '<div onclick="setColorMode(\'cluster\')" id="modeClusterBtn" '
            'style="background:#2b2b2b;color:#fff;padding:6px 12px;border-radius:20px;'
            'cursor:pointer;font-size:12px;font-weight:bold;white-space:nowrap;user-select:none;">🎨 Cluster</div>'
            '<div onclick="setColorMode(\'sentiment\')" id="modeSentBtn" '
            'style="background:#f0f0f0;color:#555;padding:6px 12px;border-radius:20px;'
            'cursor:pointer;font-size:12px;font-weight:bold;border:1px solid #ddd;'
            'white-space:nowrap;user-select:none;">😊 Sentiment</div>'
        )

    # ── JS: NODE_META is authored once in Python and never mutated in JS.
    #        showAll()/filterCluster() always reset from THIS, never from
    #        whatever the live DataSet currently happens to show — so
    #        switching clusters repeatedly, or hitting "All" after several
    #        switches, always reproduces the correct original colors.  ───────
    node_meta_json = json.dumps(node_meta)
    node_sent_json = json.dumps(node_sent)

    inject = f"""
<!-- ═══ CLUSTER TOOLBAR ═══ -->
<div id="ctoolbar" style="
  position:absolute; top:14px; left:50%; transform:translateX(-50%);
  z-index:9999;
  background:rgba(255,255,255,0.96);
  padding:8px 18px;
  border-radius:40px;
  box-shadow:0 2px 14px rgba(0,0,0,0.13);
  border:1px solid #e8e8e8;
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
  <span style="font-size:11px;font-weight:700;color:#888;letter-spacing:.08em;margin-right:4px;">ISOLATE</span>
  <div onclick="showAll()"
    style="background:#f0f0f0;color:#555;padding:6px 14px;border-radius:20px;
    cursor:pointer;font-size:12px;font-weight:bold;border:1px solid #ddd;
    white-space:nowrap;user-select:none;">↺ All</div>
  {legend_pills}
  <div onclick="exportPNG()"
    style="background:#2b2b2b;color:#fff;padding:6px 14px;border-radius:20px;
    cursor:pointer;font-size:12px;font-weight:bold;
    white-space:nowrap;user-select:none;margin-left:6px;">📷 PNG</div>
  {sent_toggle_html}
  <span style="width:1px;height:20px;background:#ddd;margin:0 2px;"></span>
  <input id="searchBox" type="text" placeholder="Search a word…"
    onkeydown="if(event.key==='Enter'){{searchWord();}}"
    style="border:1px solid #ddd;border-radius:20px;padding:6px 12px;font-size:12px;width:140px;outline:none;">
  <div onclick="searchWord()"
    style="background:#0085AF;color:#fff;padding:6px 12px;border-radius:20px;
    cursor:pointer;font-size:12px;font-weight:bold;white-space:nowrap;user-select:none;">🔍</div>
  <span id="searchMsg" style="font-size:11px;color:#C62F4B;font-weight:bold;white-space:nowrap;"></span>
</div>

<script>
// ── Immutable ground truth, authored in Python — never derived from the
//    live/rendered network, so it can never pick up a faded/highlighted
//    state by accident. ───────────────────────────────────────────────────
var NODE_META = {node_meta_json};   // {{ nodeId: {{ color, font, group }} }}
var NODE_SENT = {node_sent_json};   // same shape, sentiment-based colors (may be empty)
var ACTIVE_META = NODE_META;

function setColorMode(mode) {{
  ACTIVE_META = (mode === "sentiment" && Object.keys(NODE_SENT).length) ? NODE_SENT : NODE_META;
  var cBtn = document.getElementById("modeClusterBtn");
  var sBtn = document.getElementById("modeSentBtn");
  if (cBtn && sBtn) {{
    if (mode === "sentiment") {{
      sBtn.style.background = "#2b2b2b"; sBtn.style.color = "#fff"; sBtn.style.border = "none";
      cBtn.style.background = "#f0f0f0"; cBtn.style.color = "#555"; cBtn.style.border = "1px solid #ddd";
    }} else {{
      cBtn.style.background = "#2b2b2b"; cBtn.style.color = "#fff"; cBtn.style.border = "none";
      sBtn.style.background = "#f0f0f0"; sBtn.style.color = "#555"; sBtn.style.border = "1px solid #ddd";
    }}
  }}
  showAll();
}}

var FADE_NODE = {{ background:"rgba(220,220,220,0.25)", border:"rgba(200,200,200,0.2)",
                   highlight:{{ background:"rgba(220,220,220,0.25)", border:"rgba(200,200,200,0.2)" }} }};
var FADE_FONT = {{ color:"rgba(180,180,180,0.25)", strokeWidth:0 }};
var DIM_EDGE  = "rgba(200,200,200,0.12)";
var FULL_EDGE = "#c8d8e8";

function showAll() {{
  network.body.data.nodes.update(
    Object.keys(NODE_META).map(function(id) {{
      var m = ACTIVE_META[id];
      return {{ id:id, color:m.color, font:m.font, borderWidth:2 }};
    }})
  );
  network.body.data.edges.update(
    network.body.data.edges.get().map(function(e) {{
      return {{ id:e.id, color:{{ color:FULL_EDGE, highlight:"#FF8000" }} }};
    }})
  );
}}

function filterCluster(cid) {{
  var cs = String(cid);

  // Which node ids belong to the target cluster — from NODE_META, always.
  var inCluster = {{}};
  Object.keys(NODE_META).forEach(function(id) {{
    if (NODE_META[id].group === cs) inCluster[id] = true;
  }});

  // Every node gets an explicit, fully-specified color/font on every call —
  // selected nodes from the active metadata set (true originals), everything
  // else faded.
  network.body.data.nodes.update(
    Object.keys(NODE_META).map(function(id) {{
      var m = ACTIVE_META[id];
      if (inCluster[id]) {{
        return {{ id:id, color:m.color, font:m.font }};
      }} else {{
        return {{ id:id, color:FADE_NODE, font:FADE_FONT }};
      }}
    }})
  );

  network.body.data.edges.update(
    network.body.data.edges.get().map(function(e) {{
      var keep = inCluster[e.from] && inCluster[e.to];
      return {{ id:e.id, color:{{ color: keep ? "#7ab4c8" : DIM_EDGE, highlight:"#FF8000" }} }};
    }})
  );
}}

function normWord(s) {{ return s.replace(/_/g, " ").toLowerCase(); }}

function searchWord() {{
  var msg = document.getElementById("searchMsg");
  var q = document.getElementById("searchBox").value.trim().toLowerCase();
  msg.textContent = "";
  if (!q) return;

  var ids = Object.keys(NODE_META);
  var match = ids.find(function(id) {{ return normWord(id) === q; }});
  if (!match) {{
    match = ids.find(function(id) {{ return normWord(id).indexOf(q) !== -1; }});
  }}

  if (!match) {{
    msg.textContent = "No result found";
    return;
  }}

  showAll();
  network.selectNodes([match]);
  network.focus(match, {{
    scale: 1.5,
    animation: {{ duration: 700, easingFunction: "easeInOutQuad" }},
  }});
  network.body.data.nodes.update([{{ id: match, borderWidth: 5 }}]);
  setTimeout(function() {{
    network.body.data.nodes.update([{{ id: match, borderWidth: 2 }}]);
  }}, 1600);
}}

function exportPNG() {{
  try {{
    var canvas = network.canvas.frame.canvas;
    var link = document.createElement("a");
    link.download = "{filename}.png";
    link.href = canvas.toDataURL("image/png");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }} catch (e) {{
    alert("PNG export failed: " + e.message);
  }}
}}
</script>
"""
    # Inject just before </body>
    return html.replace("</body>", inject + "\n</body>")


# ─── Cluster bubbles (circle-packing) builder ────────────────────────────────
def build_bubbles_html(word_freq, full_partition, cluster_ids, color_map, scope,
                        size_map=None, word_sent=None, filename="cluster_bubbles"):
    """Force-directed, draggable bubble chart: each word is its own bubble,
    sized by frequency (or TF-IDF, if that weighting is active) and colored by
    cluster (or sentiment, toggle permitting), gently pulled toward its
    cluster's 'gravity well' but free to be dragged around."""
    size_map = size_map or word_freq
    word_sent = word_sent or {}
    has_sentiment = bool(word_sent)

    if scope == "Entire sample":
        scopes = list(enumerate(cluster_ids))
    else:
        idx = int(scope.split(" ")[1]) - 1
        scopes = [(idx, cluster_ids[idx])]

    nodes = []
    for i, cid in scopes:
        members = [w for w, c in full_partition.items() if c == cid]
        for w in members:
            sent = word_sent.get(w)
            nodes.append({
                "id": w,
                "name": display_label(w),
                "value": float(size_map.get(w, word_freq.get(w, 1))),
                "freq": int(word_freq.get(w, 1)),
                "cluster": cid,
                "clusterLabel": f"Cluster {i+1}",
                "sentColor": sentiment_to_color(sent) if sent is not None else "#999999",
                "sentiment": sent,
            })

    if not nodes:
        return None

    nodes_json  = json.dumps(nodes)
    colors_json = json.dumps(color_map)
    has_sent_json = json.dumps(has_sentiment)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{ margin:0; padding:0; height:100%; background:#ffffff; font-family:Arial, sans-serif; overflow:hidden; }}
  #wrap {{ position:relative; width:100%; height:100%; }}
  #viz {{ display:block; width:100%; height:100%; }}
  #toolbar {{
    position:absolute; top:12px; left:50%; transform:translateX(-50%); z-index:9999;
    background:rgba(255,255,255,0.96); padding:8px 16px; border-radius:40px;
    box-shadow:0 2px 14px rgba(0,0,0,0.13); border:1px solid #e8e8e8;
    display:flex; align-items:center; gap:8px; font-size:12px;
  }}
  #toolbar div.btn {{
    background:#2b2b2b;color:#fff;padding:6px 14px;border-radius:20px;
    cursor:pointer;font-weight:bold;white-space:nowrap;user-select:none;
  }}
  #toolbar div.btn.reset {{ background:#f0f0f0;color:#555;border:1px solid #ddd; }}
  #toolbar div.btn.inactive {{ background:#f0f0f0;color:#555;border:1px solid #ddd; }}
  .bubble-label {{ pointer-events:none; font-weight:600; fill:#fff; text-shadow:0 1px 2px rgba(0,0,0,0.4); }}
  .group-label {{ pointer-events:none; font-weight:700; fill:#555; letter-spacing:.04em; }}
  circle.word {{ cursor:grab; }}
  circle.word:active {{ cursor:grabbing; }}
  #tooltip {{
    position:absolute; z-index:10000; pointer-events:none; opacity:0;
    background:rgba(30,30,30,0.94); color:#fff; padding:8px 12px; border-radius:8px;
    font-size:12px; line-height:1.5; box-shadow:0 4px 16px rgba(0,0,0,0.25);
    transition:opacity 0.12s ease; max-width:220px;
  }}
  #tooltip b {{ font-size:13px; }}
  #tooltip .swatch {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="toolbar">
    <div class="btn reset" onclick="resetView()">↺ Reset</div>
    <div class="btn" onclick="exportPNG()">📷 PNG</div>
    <span id="sentToggleWrap"></span>
  </div>
  <div id="tooltip"></div>
  <svg id="viz"></svg>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
var NODES  = {nodes_json};
var COLORS = {colors_json};
var HAS_SENTIMENT = {has_sent_json};
var COLOR_MODE = "cluster";
// Fixed virtual canvas — NOT window.innerWidth/innerHeight. Reading window
// size at script-execution time is unreliable here: this chart can be the
// very first thing rendered inside a still-hidden/animating tab panel, in
// which case the iframe reports 0 width and the whole force layout collapses
// to nothing. A fixed coordinate space + viewBox scales visually to fill
// whatever the container turns out to be, independent of that timing.
var W = 1000, H = 640;

if (HAS_SENTIMENT) {{
  document.getElementById("sentToggleWrap").innerHTML =
    '<span style="width:1px;height:20px;background:#ddd;margin:0 2px;display:inline-block;"></span>' +
    '<div class="btn" id="modeClusterBtn" onclick="setColorMode(\\'cluster\\')">🎨 Cluster</div>' +
    '<div class="btn inactive" id="modeSentBtn" onclick="setColorMode(\\'sentiment\\')">😊 Sentiment</div>';
}}

function setColorMode(mode) {{
  COLOR_MODE = mode;
  var cBtn = document.getElementById("modeClusterBtn");
  var sBtn = document.getElementById("modeSentBtn");
  if (mode === "sentiment") {{
    sBtn.className = "btn"; cBtn.className = "btn inactive";
  }} else {{
    cBtn.className = "btn"; sBtn.className = "btn inactive";
  }}
  g.selectAll("circle.word").attr("fill", function(d) {{
    return COLOR_MODE === "sentiment" ? d.sentColor : (COLORS[d.cluster] || "#999999");
  }});
}}

var svg = d3.select("#viz")
            .attr("viewBox", [0, 0, W, H])
            .attr("preserveAspectRatio", "xMidYMid meet");
var g = svg.append("g");
var tooltip = d3.select("#tooltip");

var zoomBeh = d3.zoom().scaleExtent([0.3, 8]).on("zoom", function(ev) {{
  g.attr("transform", ev.transform);
}});
svg.call(zoomBeh);

// ── Radius scale — dynamic domain works for raw counts OR TF-IDF scores
var vals = NODES.map(function(d) {{ return d.value; }});
var minVal = d3.min(vals), maxVal = d3.max(vals);
var rScale = d3.scaleSqrt().domain([minVal, maxVal]).range([16, 58]).clamp(true);
NODES.forEach(function(d) {{ d.r = rScale(d.value); }});

// ── Cluster "gravity well" centers, spread evenly around the canvas
var clusterIds = Array.from(new Set(NODES.map(function(d) {{ return d.cluster; }})));
var centers = {{}};
if (clusterIds.length === 1) {{
  centers[clusterIds[0]] = {{ x: W / 2, y: H / 2 }};
}} else {{
  var cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.33;
  clusterIds.forEach(function(cid, i) {{
    var angle = (i / clusterIds.length) * 2 * Math.PI - Math.PI / 2;
    centers[cid] = {{ x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle) }};
  }});
}}

// ── Approximate "zone" circle behind each cluster's bubbles (static, purely
//    visual — real positions can drift slightly once dragged) ─────────────
var zoneR = {{}};
clusterIds.forEach(function(cid) {{
  var members = NODES.filter(function(d) {{ return d.cluster === cid; }});
  var area = members.reduce(function(s, d) {{ return s + d.r * d.r; }}, 0);
  zoneR[cid] = Math.sqrt(area) * 1.5 + 24;
}});

if (clusterIds.length > 1) {{
  g.selectAll("circle.zone")
    .data(clusterIds)
    .join("circle")
    .attr("class", "zone")
    .attr("cx", function(d) {{ return centers[d].x; }})
    .attr("cy", function(d) {{ return centers[d].y; }})
    .attr("r", function(d) {{ return zoneR[d]; }})
    .attr("fill", function(d) {{ return (COLORS[d] || "#999999") + "14"; }})
    .attr("stroke", function(d) {{ return (COLORS[d] || "#999999") + "55"; }})
    .attr("stroke-width", 1.5);

  g.selectAll("text.group-label")
    .data(clusterIds)
    .join("text")
    .attr("class", "group-label")
    .attr("text-anchor", "middle")
    .attr("x", function(d) {{ return centers[d].x; }})
    .attr("y", function(d) {{ return centers[d].y - zoneR[d] - 10; }})
    .style("font-size", "13px")
    .text(function(d, i) {{ return NODES.find(function(n) {{ return n.cluster === d; }}).clusterLabel; }});
}}

// ── Word bubbles ────────────────────────────────────────────────────────
var node = g.selectAll("g.node")
  .data(NODES)
  .join("g")
  .attr("class", "node");

node.append("circle")
  .attr("class", "word")
  .attr("r", function(d) {{ return d.r; }})
  .attr("fill", function(d) {{ return COLORS[d.cluster] || "#999999"; }})
  .attr("stroke", "rgba(0,0,0,0.18)")
  .attr("stroke-width", 1)
  .on("mouseenter", function(ev, d) {{
    d3.select(this).attr("stroke", "#333").attr("stroke-width", 2);
    var sentLine = (d.sentiment !== null && d.sentiment !== undefined)
      ? ("<br>Sentiment: " + (d.sentiment >= 0 ? "+" : "") + d.sentiment.toFixed(2))
      : "";
    tooltip.style("opacity", 1).html(
      '<span class="swatch" style="background:' + (COLORS[d.cluster] || "#999") + '"></span>' +
      '<b>' + d.name + '</b><br>Frequency: ' + d.freq + sentLine + '<br>' + d.clusterLabel
    );
  }})
  .on("mousemove", function(ev) {{
    var box = document.getElementById("wrap").getBoundingClientRect();
    tooltip.style("left", (ev.clientX - box.left + 16) + "px")
           .style("top",  (ev.clientY - box.top + 12) + "px");
  }})
  .on("mouseleave", function() {{
    d3.select(this).attr("stroke", "rgba(0,0,0,0.18)").attr("stroke-width", 1);
    tooltip.style("opacity", 0);
  }});

// ── Label: font-size fit to the bubble, falling back to truncation only
//    when even the smallest readable size can't fit the whole word ────────
function fitLabel(d) {{
  var usable = d.r * 1.7;
  var estCharW = 0.62;
  var size = Math.min(15, Math.max(8, usable / (d.name.length * estCharW)));
  var maxChars = Math.max(3, Math.floor(usable / (size * estCharW)));
  var text = d.name.length > maxChars ? d.name.slice(0, maxChars - 1) + "…" : d.name;
  return {{ size: size, text: text }};
}}

node.append("text")
  .attr("class", "bubble-label")
  .attr("text-anchor", "middle")
  .attr("dy", "0.32em")
  .style("font-size", function(d) {{ return fitLabel(d).size + "px"; }})
  .text(function(d) {{ return fitLabel(d).text; }});

// ── Force simulation: cluster gravity + collision + light repulsion ───────
var simulation = d3.forceSimulation(NODES)
  .force("x", d3.forceX(function(d) {{ return centers[d.cluster].x; }}).strength(0.08))
  .force("y", d3.forceY(function(d) {{ return centers[d.cluster].y; }}).strength(0.08))
  .force("collide", d3.forceCollide(function(d) {{ return d.r + 2; }}).strength(0.9))
  .force("charge", d3.forceManyBody().strength(-1))
  .on("tick", ticked);

function ticked() {{
  node.attr("transform", function(d) {{ return "translate(" + d.x + "," + d.y + ")"; }});
}}

node.call(
  d3.drag()
    .on("start", function(ev, d) {{
      if (!ev.active) simulation.alphaTarget(0.25).restart();
      d.fx = d.x; d.fy = d.y;
    }})
    .on("drag", function(ev, d) {{
      d.fx = ev.x; d.fy = ev.y;
    }})
    .on("end", function(ev, d) {{
      if (!ev.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    }})
);

function resetView() {{
  svg.transition().duration(400).call(zoomBeh.transform, d3.zoomIdentity);
  NODES.forEach(function(d) {{ d.fx = null; d.fy = null; }});
  simulation.alpha(0.6).restart();
}}

function exportPNG() {{
  var svgEl = document.getElementById("viz");
  var serializer = new XMLSerializer();
  var source = serializer.serializeToString(svgEl);
  if (!source.match(/^<svg[^>]+xmlns="http:\\/\\/www\\.w3\\.org\\/2000\\/svg"/)) {{
    source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }}
  var svgBlob = new Blob([source], {{type: "image/svg+xml;charset=utf-8"}});
  var url = URL.createObjectURL(svgBlob);
  var img = new Image();
  img.onload = function() {{
    var scale = 2;
    var canvas = document.createElement("canvas");
    canvas.width = W * scale;
    canvas.height = H * scale;
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, W, H);
    URL.revokeObjectURL(url);
    var link = document.createElement("a");
    link.download = "{filename}.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  }};
  img.src = url;
}}
</script>
</body>
</html>
"""
    return html


# ─── Sidebar ─────────────────────────────────────────────────────────────────
if "hide_frag_warning" not in st.session_state:
    st.session_state["hide_frag_warning"] = False

with st.sidebar:
    if not st.session_state["hide_frag_warning"]:
        warn_col, close_col = st.columns([10, 1])
        warn_col.markdown(
            """<div style="background:#fff3cd;color:#664d03;border:1px solid #ffe69c;
            border-radius:8px;padding:7px 10px;font-size:11.5px;line-height:1.4;">
            ⚠️ This verbatim analysis suite is designed for <b>non-fragrance-related</b> text,
            or <b>huge batches of aggregated verbatims</b> (e.g. thousands). For fragrance /
            fragrance test analysis, please use <b>Text-Mining</b> or <b>Verbatim Studio</b> instead.
            </div>""",
            unsafe_allow_html=True,
        )
        if close_col.button("✕", key="dismiss_frag_warning", help="Dismiss this notice"):
            st.session_state["hide_frag_warning"] = True
            st.rerun()

    st.title("⚙️ Settings")
    lang_choice = st.selectbox("🌍 Verbatim language", list(LANGUAGES.keys()), index=0, key="lang_choice")
    lang_info = LANGUAGES[lang_choice]
    if lang_info["spacy_model"] and load_spacy_model(lang_info["spacy_model"]) is None:
        st.caption(
            f"⚠️ spaCy model **{lang_info['spacy_model']}** isn't installed — {lang_choice} will fall back "
            "to basic tokenization without proper lemmatization until it's added (see requirements.txt)."
        )
    if lang_choice != "English":
        st.caption("ℹ️ Sentiment analysis is English-only and is disabled for this language.")
    uploaded_file = st.file_uploader("📂 Upload Excel corpus", type=["xlsx"])
    st.markdown("---")
    min_freq   = st.slider("Min word occurrences",         1, 50,  5)
    min_edge   = st.slider(
        "Min connection strength", 1, 20, 3,
        help="How many verbatims two words must co-occur in together before "
             "a line is drawn between them on the map. Higher = fewer, "
             "stronger connections shown (a sparser, cleaner graph); "
             "lower = more connections shown, including weaker/noisier ones.",
    )
    n_clusters = st.slider(
        "Target number of clusters", 2, 10, 5,
        help="Best-effort target, not an exact count. Clustering can raise the "
             "number of groups fairly freely, but it can never merge two fully "
             "disconnected sets of words into one cluster — so if your filters "
             "split the graph into several separate 'islands' with no "
             "connections between them, that number of islands becomes a hard "
             "floor no matter how low you set this. Lowering 'Min connection "
             "strength' or 'Min word occurrences' usually helps reach a lower "
             "target, since it lets more words connect to each other.",
    )
    st.markdown("---")
    st.caption("🔤 Phrases & weighting")
    use_phrases = st.checkbox("Detect common phrases (bigrams)", value=True, key="use_phrases")
    min_bigram_freq = st.slider(
        "Min phrase occurrences", 2, 20, 4, key="min_bigram_freq", disabled=not use_phrases
    )
    use_tfidf = st.checkbox(
        "Use TF-IDF weighting instead of raw frequency", value=False, key="use_tfidf",
        help="Sizes words/bubbles/nodes by how distinctive they are across verbatims, "
             "instead of by raw occurrence count.",
    )
    use_spellcheck = st.checkbox(
        "✏️ Correct spelling before analysis (beta)", value=False, key="use_spellcheck",
        help="Fixes likely typos using a general-purpose dictionary before tokenizing "
             "— only single-character-edit fixes (conservative), e.g. 'frehs' → "
             "'fresh'. Only affects the analysis (tokens, clustering, sentiment); the "
             "original verbatim text is never changed in exports or respondent "
             "drill-down. It can occasionally 'correct' a legitimate niche or "
             "brand-specific word it doesn't recognize — a full list of every change "
             "made appears after you generate the map, so you can review it. Not "
             "guaranteed to be available for every verbatim language.",
    )
    st.markdown("---")
    st.caption("ℹ️ After adding words here, click **Generate map** again to regenerate the analysis for newly excluded words.")
    user_extra_stops = st.text_area("Extra exclusion words (comma-sep):", "")
    st.markdown("---")
    st.caption("🎯 Sub-corpus filter (optional)")
    subcorpus_input = st.text_area(
        "Only analyze verbatims containing at least one of these words/phrases:",
        "",
        key="subcorpus_input",
        placeholder="e.g. fresh, clean",
        help="Leave empty to use the full corpus. Matches on the ORIGINAL verbatim "
             "text — before lemmatization — so it means the same thing regardless "
             "of language and doesn't depend on guessing a word's lemma form. "
             "Whole-word, case-insensitive match: 'clean' matches 'very clean now' "
             "but not inside 'uncleaned'. A verbatim is kept if it contains at "
             "least one of the words/phrases you list (comma-separated) — not all "
             "of them. Regenerate the map after changing this.",
    )
    subcorpus_words = [w.strip().lower() for w in subcorpus_input.split(",") if w.strip()]

all_stops = set(
    DEFAULT_EXCLUSIONS_BY_LANG.get(lang_choice, [])
    + [w.strip().lower() for w in user_extra_stops.split(",") if w.strip()]
)

# ─── Main ────────────────────────────────────────────────────────────────────
st.title("🌐 Semantic Relationship Map")

if uploaded_file:
    df  = pd.read_excel(uploaded_file)
    col = st.selectbox("Text column", df.columns)

    if subcorpus_words:
        preview_mask = build_subcorpus_mask(df[col].tolist(), subcorpus_words)
        st.caption(
            f"🎯 Sub-corpus filter active — **{sum(preview_mask)} of {len(df)}** verbatims match "
            f"({', '.join(subcorpus_words)}). The analysis will run on this subset only."
        )

    # Placed via st.sidebar so it renders in the sidebar even though this
    # code runs in the main body — it needs df.columns, which isn't known
    # until a file is uploaded.
    st.sidebar.markdown("---")
    st.sidebar.caption("🔀 Group comparison")
    group_col_choice = st.sidebar.selectbox(
        "Compare groups by (optional)",
        ["None"] + [c for c in df.columns if c != col],
        key="group_col_choice",
    )

    if st.button("🚀 Generate map", use_container_width=True):
        lemmatizer_en = load_lemmatizer()
        spacy_nlp = load_spacy_model(lang_info["spacy_model"])
        stop_words = load_stopwords(lang_info["nltk_stop"])
        sentiment_analyzer = load_sentiment_analyzer() if lang_info["code"] == "en" else None

        with st.spinner("Analysing text and building graph…"):

            # ── Sub-corpus filter (optional) — applied FIRST, on the raw
            #    text, before any tokenization/lemmatization. Everything
            #    downstream (tokens, sentiment, graph, clustering, group
            #    comparison, respondent drill-down) then only ever sees this
            #    filtered subset. ─────────────────────────────────────────────
            n_before_filter = len(df)
            if subcorpus_words:
                mask = build_subcorpus_mask(df[col].tolist(), subcorpus_words)
                df = df.loc[mask].reset_index(drop=True)
                if df.empty:
                    st.warning(
                        "No verbatims match your sub-corpus filter words — try different "
                        "words/phrases, or clear the filter to use the full corpus."
                    )
                    st.stop()

            # ── Spelling correction (optional) — corrects only the ANALYSIS
            #    input, never the original verbatim column, so respondent
            #    drill-down and exports always show exactly what people wrote.
            analysis_col = col
            spelling_corrections, spelling_corrected_occurrences = {}, 0
            if use_spellcheck:
                spellchecker = load_spellchecker(lang_info["code"])
                if spellchecker is None:
                    st.warning(f"Spelling correction isn't available for {lang_choice} — skipped.")
                else:
                    spelling_corrections, spelling_corrected_occurrences = build_spelling_corrections(
                        df[col].tolist(), spellchecker
                    )
                    if spelling_corrections:
                        df["_analysis_text"] = df[col].apply(
                            lambda t: apply_spelling_corrections(t, spelling_corrections)
                        )
                        analysis_col = "_analysis_text"

            # Tokenise
            df["tokens"] = df[analysis_col].apply(
                lambda x: preprocess(x, lang_info["code"], lemmatizer_en, spacy_nlp, all_stops, stop_words)
            )

            # ── Phrase detection: merge recurring adjacent word pairs into
            #    single phrase tokens ("easy_apply") before anything downstream
            #    counts/clusters/graphs them. ─────────────────────────────────
            if use_phrases:
                top_bigrams = extract_top_bigrams(df["tokens"].tolist(), min_freq=min_bigram_freq, top_n=30)
                if top_bigrams:
                    df["tokens"] = df["tokens"].apply(lambda toks: merge_bigrams(toks, top_bigrams))

            # ── Sentiment: English only — VADER's lexicon is English words/
            #    emoticons/intensifiers, so it would silently return
            #    near-meaningless near-zero scores for other languages rather
            #    than erroring, which is worse than just not showing it. ─────
            if sentiment_analyzer is not None:
                row_sentiments = compute_row_sentiment(df[analysis_col].tolist(), sentiment_analyzer)
                word_sent = compute_word_sentiment(df["tokens"].tolist(), row_sentiments)
            else:
                word_sent = {}

            # Frequencies
            word_freq   = Counter(itertools.chain.from_iterable(df["tokens"]))
            pair_counts = Counter()
            for tokens in df["tokens"]:
                ut = sorted(set(tokens))
                for pair in itertools.combinations(ut, 2):
                    pair_counts[pair] += 1

            # Build graph
            G = nx.Graph()
            for (u, v), w in pair_counts.items():
                if w >= min_edge and word_freq[u] >= min_freq and word_freq[v] >= min_freq:
                    G.add_node(u, size=word_freq[u])
                    G.add_node(v, size=word_freq[v])
                    G.add_edge(u, v, weight=w)

            if len(G.nodes) == 0:
                st.warning("No connections found. Try lowering the sliders.")
                st.stop()

            # ── TF-IDF weighting (optional) — used only for VISUAL SIZING
            #    (node/bubble size, word cloud weight); graph edges/thresholds
            #    still use raw occurrence counts so "Min word occurrences"
            #    keeps meaning what it says. ──────────────────────────────────
            if use_tfidf:
                tfidf_scores = compute_tfidf_scores(df["tokens"].tolist())
                size_map = {n: tfidf_scores.get(n, 0.0001) for n in G.nodes()}
            else:
                size_map = {n: word_freq[n] for n in G.nodes()}

            # ── Louvain clustering — resolution search, not seed search
            #    (see cluster_to_target docstring for why). ──────────────────
            best_p, best_d, n_components = cluster_to_target(G, n_clusters)

            # ── Spring layout → fixed pixel coords ─────────────────────────
            pos = nx.spring_layout(G, seed=42, k=3.5 / max(1, len(G.nodes) ** 0.5))
            for node, (x, y) in pos.items():
                G.nodes[node]["x"] = float(x) * 1000
                G.nodes[node]["y"] = float(y) * 1000

            cluster_ids = sorted(set(best_p.values()))

            # ── Group comparison data (optional) ────────────────────────────
            group_freqs, group_counts, group_col = None, None, None
            if group_col_choice != "None":
                group_col = group_col_choice
                group_freqs, group_counts = {}, {}
                for val, sub in df.groupby(group_col_choice):
                    toks = list(itertools.chain.from_iterable(sub["tokens"]))
                    group_freqs[str(val)] = Counter(toks)
                    group_counts[str(val)] = len(sub)

            # ── Respondent-level lookup table (kept minimal) ────────────────
            keep_cols = [col, "tokens"] + ([group_col_choice] if group_col_choice != "None" else [])
            resp_df = df[keep_cols].copy()

            # ── Everything downstream (map html, word cloud, bubbles) only
            #    needs these — cache them and nothing more. The map/bubbles
            #    HTML is built at RENDER time (below), not here, because it
            #    depends on the current cluster color_map — which the color
            #    pickers can change on later reruns without needing a fresh
            #    "Generate map" click. ─────────────────────────────────────
            st.session_state["results"] = {
                "word_freq": word_freq,
                "size_map": size_map,
                "word_sent": word_sent,
                "G": G,
                "best_p": best_p,
                "cluster_ids": cluster_ids,
                "group_freqs": group_freqs,
                "group_counts": group_counts,
                "group_col": group_col,
                "resp_df": resp_df,
                "text_col": col,
                "using_tfidf": use_tfidf,
                "target_n_clusters": n_clusters,
                "cluster_match_diff": best_d,
                "n_components": n_components,
                "n_before_filter": n_before_filter,
                "n_after_filter": len(df),
                "subcorpus_words": list(subcorpus_words),
                "spelling_corrections": spelling_corrections,
                "spelling_corrected_occurrences": spelling_corrected_occurrences,
            }
            # New analysis → reset custom cluster colors to the default
            # palette (a prior custom pick may not even make sense if the
            # number/order of clusters changed).
            st.session_state["cluster_colors"] = {
                cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cid in enumerate(cluster_ids)
            }

# ── Render from session_state — survives selectbox/slider/color-picker reruns
if "results" in st.session_state:
    res = st.session_state["results"]
    word_freq    = res["word_freq"]
    size_map     = res["size_map"]
    word_sent    = res["word_sent"]
    G            = res["G"]
    best_p       = res["best_p"]
    cluster_ids  = res["cluster_ids"]
    group_freqs  = res.get("group_freqs")
    group_counts = res.get("group_counts")
    group_col    = res.get("group_col")
    resp_df      = res.get("resp_df")
    text_col     = res.get("text_col")
    using_tfidf  = res.get("using_tfidf", False)
    target_n_clusters  = res.get("target_n_clusters")
    cluster_match_diff = res.get("cluster_match_diff", 0)
    n_components       = res.get("n_components")
    n_before_filter    = res.get("n_before_filter")
    n_after_filter     = res.get("n_after_filter")
    subcorpus_words    = res.get("subcorpus_words") or []
    spelling_corrections = res.get("spelling_corrections") or {}
    spelling_corrected_occurrences = res.get("spelling_corrected_occurrences", 0)

    # ── Custom cluster colors ────────────────────────────────────────────────
    with st.expander("🎨 Cluster colors", expanded=False):
        st.caption("Pick a color per cluster — the map, cluster bubbles, and word cloud all update to match.")
        picker_cols = st.columns(len(cluster_ids))
        for i, cid in enumerate(cluster_ids):
            top_word = max(
                (w for w, c in best_p.items() if c == cid),
                key=lambda w: word_freq[w],
                default=f"Cluster {i+1}",
            )
            picked = picker_cols[i].color_picker(
                f"C{i+1} · {display_label(top_word)}",
                value=st.session_state["cluster_colors"].get(cid, CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
                key=f"color_cluster_{cid}",
            )
            st.session_state["cluster_colors"][cid] = picked

    color_map = st.session_state["cluster_colors"]
    html_map = build_html(
        G, best_p, word_freq, color_map,
        size_map=size_map, word_sent=word_sent, filename="semantic_map",
    )

    # ── Cluster summary cards ───────────────────────────────────────────────
    st.markdown("### Cluster overview")
    if spelling_corrections:
        with st.expander(
            f"✏️ Spelling correction applied — {len(spelling_corrections)} unique words changed "
            f"({spelling_corrected_occurrences} occurrences). Click to review."
        ):
            st.caption(
                "Only affects the analysis (tokenizing, clustering, sentiment) — the original "
                "verbatim text is unchanged everywhere else (respondent drill-down, exports)."
            )
            correction_rows = sorted(spelling_corrections.items())
            st.dataframe(
                pd.DataFrame(correction_rows, columns=["Original", "Corrected"]),
                use_container_width=True, height=min(300, 40 + 35 * len(correction_rows)),
            )
    if subcorpus_words:
        st.caption(
            f"🎯 Sub-corpus filter applied — analyzing **{n_after_filter} of {n_before_filter}** verbatims "
            f"matching: {', '.join(subcorpus_words)}."
        )
    if target_n_clusters and cluster_match_diff:
        if n_components and target_n_clusters < n_components:
            st.caption(
                f"Found **{len(cluster_ids)} clusters** — with the current filters, this graph splits into "
                f"**{n_components} separate, disconnected groups of words** with no links between them at all. "
                f"Clustering can never merge fully disconnected groups into one cluster (there's nothing to base "
                f"a merge on), so {n_components} is a hard floor here — you can't go below it just by raising "
                "the target. Lower 'Min connection strength' or 'Min word occurrences' to let more words connect "
                "to each other, which can reduce the number of disconnected groups."
            )
        else:
            st.caption(
                f"Found **{len(cluster_ids)} clusters** — the closest achievable match to your target of "
                f"{target_n_clusters} for this graph (Louvain community detection can't be forced to an exact "
                "count). Try adjusting 'Min connection strength' or 'Min word occurrences' too — they reshape "
                "the graph itself and can shift how many natural clusters it settles into."
            )
    if using_tfidf:
        st.caption("Sizing by TF-IDF weight (enabled in the sidebar) — words distinctive to a cluster stand out, not just frequent ones.")
    card_cols = st.columns(len(cluster_ids))
    for i, cid in enumerate(cluster_ids):
        members = sorted(
            [w for w, c in best_p.items() if c == cid],
            key=lambda w: -word_freq[w],
        )
        col_bg = color_map[cid]
        sent = cluster_avg_sentiment(members, word_freq, word_sent)
        if sent is None:
            sent_badge = ""
        elif sent > 0.05:
            sent_badge = f"🙂 +{sent:.2f}"
        elif sent < -0.05:
            sent_badge = f"🙁 {sent:.2f}"
        else:
            sent_badge = f"😐 {sent:+.2f}"
        card_cols[i].markdown(
            f"""<div style="background:{col_bg};color:#fff;padding:12px 10px;
                border-radius:10px;border-left:5px solid rgba(0,0,0,0.2);">
                <div style="font-size:.75em;opacity:.8;letter-spacing:.06em;">CLUSTER {i+1}</div>
                <div style="font-weight:bold;font-size:1.05em;margin:4px 0;">
                  {display_label(members[0]).upper() if members else "—"}
                </div>
                <div style="font-size:.72em;line-height:1.4;opacity:.9;">
                  {", ".join(display_label(w) for w in members[1:5])}{"…" if len(members) > 5 else ""}
                </div>
                <div style="font-size:.7em;margin-top:6px;opacity:.75;">
                  {len(members)} words {("· " + sent_badge) if sent_badge else ""}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Download button in sidebar — now stable across reruns since html_map
    # comes from session_state rather than only existing mid-button-click.
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        "💾 Download HTML map",
        data=html_map,
        file_name="semantic_map.html",
        mime="text/html",
        use_container_width=True,
        key="download_map_html",
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_labels = ["🌐 Interactive Map", "🔵 Cluster Bubbles", "☁️ Word Cloud", "🔬 Drill-Down"]
    if group_freqs:
        tab_labels.append("🔀 Group Comparison")
    tabs = st.tabs(tab_labels)

    # ── Tab 1: Interactive Map + Sentiment by Cluster ───────────────────────
    with tabs[0]:
        st.components.v1.html(html_map, height=750, scrolling=False)

        if word_sent:
            st.markdown("### 😊 Sentiment by Cluster")
            st.caption("Average VADER sentiment of respondent rows mentioning each cluster's words (−1 = negative, +1 = positive).")
            sent_rows = []
            for i, cid in enumerate(cluster_ids):
                members = [w for w, c in best_p.items() if c == cid]
                s = cluster_avg_sentiment(members, word_freq, word_sent)
                sent_rows.append((f"C{i+1} · {display_label(max(members, key=lambda w: word_freq[w]))}" if members else f"C{i+1}", s or 0.0))
            fig, ax = plt.subplots(figsize=(9, max(1.6, 0.5 * len(sent_rows))))
            labels = [r[0] for r in sent_rows]
            scores = [r[1] for r in sent_rows]
            bar_colors = [sentiment_to_color(s) for s in scores]
            ax.barh(labels, scores, color=bar_colors)
            ax.axvline(0, color="#333", linewidth=0.8)
            ax.set_xlim(-1, 1)
            ax.set_xlabel("Average sentiment")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 2: Cluster Bubbles ────────────────────────────────────────────────
    with tabs[1]:
        st.caption("Word size = frequency (or TF-IDF, if enabled) · color = cluster · drag a bubble to move it · scroll/drag background to zoom & pan")
        bubble_options = ["Entire sample"] + [f"Cluster {i+1}" for i in range(len(cluster_ids))]
        bubble_scope = st.selectbox("Show bubbles for:", bubble_options, key="bubble_scope")

        bubble_fname = (
            "cluster_bubbles_all" if bubble_scope == "Entire sample"
            else f"cluster_bubbles_{bubble_scope.replace(' ', '_').lower()}"
        )
        bubble_html = build_bubbles_html(
            word_freq, best_p, cluster_ids, color_map, bubble_scope,
            size_map=size_map, word_sent=word_sent, filename=bubble_fname,
        )
        if bubble_html:
            st.components.v1.html(bubble_html, height=650, scrolling=False)
            st.download_button(
                "💾 Download cluster bubbles (HTML)",
                data=bubble_html,
                file_name=f"{bubble_fname}.html",
                mime="text/html",
                use_container_width=True,
                key="download_bubbles_html",
            )
        else:
            st.info("No words to display for this selection.")

    # ── Tab 3: Word Cloud ────────────────────────────────────────────────────
    with tabs[2]:
        wc_col1, wc_col2 = st.columns([2, 1])
        cloud_options = ["Entire sample"] + [f"Cluster {i+1}" for i in range(len(cluster_ids))]
        cloud_scope = wc_col1.selectbox("Show word cloud for:", cloud_options, key="cloud_scope")
        font_choice = wc_col2.selectbox("Font", list(FONT_OPTIONS.keys()), key="cloud_font")

        if cloud_scope == "Entire sample":
            cloud_freqs = {w: size_map.get(w, word_freq[w]) for w in word_freq}
            focus_cid = None
        else:
            idx = int(cloud_scope.split(" ")[1]) - 1
            focus_cid = cluster_ids[idx]
            cloud_freqs = {w: size_map.get(w, word_freq[w]) for w, c in best_p.items() if c == focus_cid}

        if cloud_freqs:
            font_path = FONT_OPTIONS[font_choice]
            if font_path and not os.path.exists(font_path):
                st.caption(f"⚠️ '{font_choice}' font file not found at `{font_path}` — using the default font instead. Add the .ttf there to enable it.")
                font_path = None

            # WordCloud needs the words it displays as keys — swap in display
            # labels (spaces instead of underscores for phrase tokens) here, and
            # keep a reverse lookup so the color function can still find each
            # word's original cluster/frequency data.
            display_freqs = {display_label(w): v for w, v in cloud_freqs.items()}
            disp_to_orig = {display_label(w): w for w in cloud_freqs}

            wc_kwargs = dict(
                width=1100, height=550, background_color="white", prefer_horizontal=0.9,
            )
            if font_path:
                wc_kwargs["font_path"] = font_path

            wc = WordCloud(**wc_kwargs).generate_from_frequencies(display_freqs)

            # ── Recolor to match cluster colors, consistent with the map/bubbles.
            if focus_cid is None:
                # Entire sample: solid color per word's own cluster.
                def _color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                    orig = disp_to_orig.get(word, word)
                    cid = best_p.get(orig)
                    return rgb_str(color_map.get(cid, "#999999"))
            else:
                # Single cluster focus: shades of that one cluster's color,
                # darker for more frequent/weighted words — keeps everything
                # readably within the cluster's hue instead of introducing new
                # colors, while relative importance is still visible at a glance.
                base_color = color_map[focus_cid]
                freqs = list(cloud_freqs.values())
                fmin, fmax = min(freqs), max(freqs)
                def _color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                    orig = disp_to_orig.get(word, word)
                    f = cloud_freqs.get(orig, fmin)
                    t = 0.5 if fmax == fmin else (f - fmin) / (fmax - fmin)
                    return shade_rgb_str(base_color, 0.3 + t * 0.55)

            wc.recolor(color_func=_color_func, random_state=42)

            fig, ax = plt.subplots(figsize=(11, 5.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

            buf = io.BytesIO()
            wc.to_image().save(buf, format="PNG")
            st.download_button(
                "💾 Download word cloud (PNG)",
                data=buf.getvalue(),
                file_name=f"wordcloud_{cloud_scope.replace(' ', '_').lower()}.png",
                mime="image/png",
                use_container_width=True,
                key="download_wordcloud_png",
            )
        else:
            st.info("No words to display for this selection.")

    # ── Tab 4: Hierarchical Drill-Down + Respondent-level Drill-down ────────
    with tabs[3]:
        st.markdown("### 🔬 Hierarchical Drill-Down")
        st.caption("Zoom into one cluster and re-run clustering on just its words to reveal sub-structure.")
        drill_options = ["None"] + [f"Cluster {i+1}" for i in range(len(cluster_ids))]
        drill_choice = st.selectbox("Drill into:", drill_options, key="drill_choice")

        if drill_choice != "None":
            idx = int(drill_choice.split(" ")[1]) - 1
            cid = cluster_ids[idx]
            members = [w for w, c in best_p.items() if c == cid]
            subG = G.subgraph(members).copy()

            if subG.number_of_nodes() < 4 or subG.number_of_edges() < 2:
                st.info("Not enough internal structure in this cluster to sub-divide.")
            else:
                sub_partition = community_louvain.best_partition(subG, random_state=0)
                sub_ids = sorted(set(sub_partition.values()))
                sub_color_map = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(sub_ids)}

                st.write(f"**{drill_choice}** split into {len(sub_ids)} sub-groups:")
                sub_cols = st.columns(len(sub_ids))
                for i, scid in enumerate(sub_ids):
                    smembers = sorted([w for w, c in sub_partition.items() if c == scid], key=lambda w: -word_freq.get(w, 0))
                    sub_cols[i].markdown(
                        f"""<div style="background:{sub_color_map[scid]};color:#fff;padding:8px;border-radius:8px;font-size:.75em;">
                        <b>{display_label(smembers[0]).upper() if smembers else '—'}</b><br>{", ".join(display_label(w) for w in smembers[1:5])}
                        </div>""",
                        unsafe_allow_html=True,
                    )

                sub_html = build_html(
                    subG, sub_partition, word_freq, sub_color_map,
                    size_map=size_map, word_sent=word_sent,
                    filename=f"drilldown_{drill_choice.replace(' ', '_').lower()}",
                )
                st.components.v1.html(sub_html, height=520, scrolling=False)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 🔎 Respondent-level Drill-down")
        if resp_df is not None and text_col is not None:
            all_words = sorted(G.nodes(), key=lambda w: -word_freq.get(w, 0))
            pick_word = st.selectbox(
                "Show verbatims containing:",
                all_words,
                format_func=display_label,
                key="drilldown_word",
            )
            mask = resp_df["tokens"].apply(lambda toks: pick_word in toks)
            matches = resp_df.loc[mask]
            st.write(f"**{len(matches)}** respondent(s) mention *{display_label(pick_word)}*")
            show_cols = [text_col] + ([group_col] if group_col else [])
            st.dataframe(matches[show_cols], use_container_width=True, height=300)
            if len(matches):
                csv_buf = matches[show_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Download matching verbatims (CSV)",
                    data=csv_buf,
                    file_name=f"verbatims_{pick_word}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_verbatims_csv",
                )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 🔗 Word Co-occurrence Table")
        edge_rows = [
            {"word_1": display_label(u), "word_2": display_label(v), "co_occurrences": d.get("weight", 1)}
            for u, v, d in G.edges(data=True)
        ]
        edge_df = pd.DataFrame(edge_rows).sort_values("co_occurrences", ascending=False).reset_index(drop=True)
        st.dataframe(edge_df, use_container_width=True, height=300)

        freq_df = pd.DataFrame(
            [{"word": display_label(w), "frequency": word_freq[w], "cluster": best_p[w] + 1} for w in G.nodes()]
        ).sort_values("frequency", ascending=False).reset_index(drop=True)

        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            edge_df.to_excel(writer, index=False, sheet_name="co_occurrences")
            freq_df.to_excel(writer, index=False, sheet_name="word_frequency")
        st.download_button(
            "💾 Download co-occurrence data (XLSX)",
            data=xlsx_buf.getvalue(),
            file_name="word_cooccurrence.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_cooc_xlsx",
        )

    # ── Tab 5: Group Comparison & Diff View (only if a group column is set) ─
    if group_freqs:
        with tabs[4]:
            gnames = list(group_freqs.keys())
            if len(gnames) < 2:
                st.info(f"Only one group value found in '{group_col}' — need at least two to compare.")
            else:
                gc1, gc2 = st.columns(2)
                group_a = gc1.selectbox("Group A", gnames, index=0, key="diff_group_a")
                default_b_idx = 1 if len(gnames) > 1 else 0
                group_b = gc2.selectbox("Group B", gnames, index=default_b_idx, key="diff_group_b")

                if group_a == group_b:
                    st.info("Pick two different groups to compare.")
                else:
                    freqs_a, freqs_b = group_freqs[group_a], group_freqs[group_b]
                    total_a = sum(freqs_a.values()) or 1
                    total_b = sum(freqs_b.values()) or 1
                    eps = 0.5
                    diffs = []
                    for w in set(freqs_a) | set(freqs_b):
                        a, b = freqs_a.get(w, 0), freqs_b.get(w, 0)
                        if a + b < 3:
                            continue
                        rate_a = (a + eps) / (total_a + eps)
                        rate_b = (b + eps) / (total_b + eps)
                        diffs.append((w, math.log2(rate_a / rate_b), a, b))
                    diffs.sort(key=lambda x: x[1])

                    if not diffs:
                        st.info("Not enough overlapping vocabulary between these two groups to compare.")
                    else:
                        top_b = diffs[:12]
                        top_a = list(reversed(diffs[-12:]))
                        plot_rows = top_a + top_b
                        labels = [display_label(w) for w, _, _, _ in plot_rows]
                        scores = [s for _, s, _, _ in plot_rows]
                        bar_colors = ["#0085AF" if s > 0 else "#C62F4B" for s in scores]

                        fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(plot_rows))))
                        y = list(range(len(plot_rows)))
                        ax.barh(y, scores, color=bar_colors)
                        ax.set_yticks(y)
                        ax.set_yticklabels(labels, fontsize=9)
                        ax.invert_yaxis()
                        ax.axvline(0, color="#333", linewidth=0.8)
                        ax.set_xlabel(f"← more typical of {group_b}     |     more typical of {group_a} →")
                        ax.set_title(f"Word usage skew: {group_a} vs {group_b}")
                        st.pyplot(fig)
                        plt.close(fig)

                    wc_a, wc_b = st.columns(2)
                    wc_a.markdown(f"**Top words — {group_a}** ({group_counts[group_a]} rows)")
                    wc_a.dataframe(
                        pd.DataFrame(freqs_a.most_common(10), columns=["word", "count"]).assign(word=lambda d: d["word"].map(display_label)),
                        hide_index=True, use_container_width=True,
                    )
                    wc_b.markdown(f"**Top words — {group_b}** ({group_counts[group_b]} rows)")
                    wc_b.dataframe(
                        pd.DataFrame(freqs_b.most_common(10), columns=["word", "count"]).assign(word=lambda d: d["word"].map(display_label)),
                        hide_index=True, use_container_width=True,
                    )
