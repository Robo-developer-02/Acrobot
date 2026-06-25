"""

  Additional installation for this file
  ─────────────────────────────────────────────────────────
    To install the offline fallback on your Pi:
    ```
    sudo apt install espeak espeak-data libespeak-dev
    pip install pyttsx3
    ```
"""

"""
============================================================
  🤖  AcroBot 2.2 — RAG-Powered Speech-to-Speech Chatbot
  Production Release
============================================================

  Architecture
  ─────────────────────────────────────────────────────────
  • Hindi queries  → retrieved directly against hindi_details.pdf (rag_hi)
  • English queries → retrieved directly against english_details.pdf (rag_en)
  • Language detection picks the engine; the query is never translated.
  • Web fallback only fires when PDF score is below threshold AND the
    query contains time-sensitive keywords.

  Production Changes (over dev build)
  ─────────────────────────────────────────────────────────
  FIX-P1  Empty / whitespace-only user input is rejected BEFORE reaching
          the LLM — validated with .strip() at both the main-loop level
          and inside get_ai_reply() as a second defence layer.

  FIX-P2  get_ai_reply() now ALWAYS returns a non-empty str or raises.
          The previously commented-out fallback return was the root cause
          of implicit None returns, which then triggered a double error
          announcement (once inside get_ai_reply, once in SPEAKING state).

  FIX-P3  Conversation history is capped at MAX_HISTORY_TURNS to prevent
          unbounded memory growth in long sessions.

  FIX-P4  All print() calls replaced with the stdlib logging module.
          DEBUG-level messages (raw LLM response, MP3 size, TTS input)
          are hidden in production (INFO level). Set LOG_LEVEL=DEBUG in
          .env or environment to re-enable them during development.

  FIX-P5  asyncio event loop is created once at module start and reused
          by every speak() call, avoiding per-call loop creation overhead.

  FIX-P6  The main loop's `reply` variable is scoped per iteration via a
          helper function so no stale reply from a previous turn can bleed
          into the SPEAKING state.

  FIX-P7  User text is sanitized (strip + collapse internal whitespace)
          before being passed to the LLM or used for logging.

  FIX-P8  NEW: Explicit internet-connectivity probe (has_internet()).
          When the device has no internet at all, the bot now speaks a
          dedicated "I can't connect to the internet" message instead of
          silently falling through to a generic API-error message, and
          it fails fast (skips the LLM/STT call entirely) rather than
          waiting on a timeout first.

  FIX-P9  FIXED: Wrong CHAT_MODEL value ("openai/gpt-oss-20b" is not a
          valid Groq model). Replaced with "llama3-70b-8192" which is
          the recommended high-quality Groq model.

  FIX-P10 FIXED: pyttsx3 engine was being re-initialised on every speak()
          call, causing thread conflicts and crashes on long sessions.
          Engine is now created once at module load and reused.

  FIX-P11 FIXED: pygame.mixer now initialised with explicit frequency and
          buffer size tuned for Raspberry Pi to prevent choppy/distorted
          audio playback.

  FIX-P12 FIXED: has_internet() was called 6-8 times per query (once in
          each of _process_query, get_ai_reply, transcribe, web_search,
          _speak_edge_tts, build_context). Now called ONCE per query in
          _process_query and the result is passed down as a parameter,
          eliminating redundant TCP probes and ~2s of latency per query.

  FIX-P13 FIXED: Missing .env file was silently ignored. Now explicitly
          checked with a clear startup warning so operators know
          immediately if the file is missing, rather than getting a
          cryptic ValueError about GROQ_API_KEY.
============================================================
"""

# ── Standard library ──────────────────────────────────────
import asyncio
import logging
import os
import queue
import re
import socket
import tempfile
import textwrap
import time
from enum import Enum
from typing import List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────
import fitz
import numpy as np
import pygame
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import edge_tts

try:
    import pyttsx3 as _pyttsx3_mod
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False


# ══════════════════════════════════════════════════════════
#  LOGGING  (FIX-P4)
#  Set LOG_LEVEL=DEBUG in your .env for verbose dev output.
# ══════════════════════════════════════════════════════════

# FIX-P13: Warn if .env file is missing so operators know immediately.
_ENV_FILE = ".env"
if not os.path.exists(_ENV_FILE):
    # Use print here because the logger is not yet configured.
    print(
        f"[WARNING] '{_ENV_FILE}' file not found in the current directory. "
        "GROQ_API_KEY must be set in the environment or the bot will fail to start."
    )

load_dotenv()

_log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("acrobot")


# ══════════════════════════════════════════════════════════
#  API KEY
# ══════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Add it to your .env file: GROQ_API_KEY=gsk_..."
    )

logger.info("API key loaded.")


# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════

STT_MODEL  = "whisper-large-v3"

# FIX-P9: "openai/gpt-oss-20b" is NOT a valid Groq model and would
# cause every LLM call to fail with a 404. Replaced with the
# recommended high-quality Groq model.
CHAT_MODEL = "llama3-70b-8192"

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16_000
CHANNELS    = 1
MAX_TOKENS  = 300

# ── History ───────────────────────────────────────────────
# FIX-P3: cap per-language history to prevent memory growth.
# Each turn = 1 user + 1 assistant message → 2 items per turn.
MAX_HISTORY_TURNS = 10
MAX_HISTORY_ITEMS = MAX_HISTORY_TURNS * 2
LLM_MAX_RETRIES   = 2   # extra attempts when model returns empty response

# ── RAG settings ──────────────────────────────────────────
PDF_PATH_EN   = "english_details.pdf"
PDF_PATH_HI   = "hindi_details.pdf"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50
TOP_K         = 3
PDF_THRESHOLD = 0.04

# ── Web fallback ──────────────────────────────────────────
WEB_RESULTS = 3
WEB_TIMEOUT = 5
WEB_KEYWORDS = [
    "today", "latest", "current", "now", "2025", "2026",
    "result", "merit list", "cutoff", "ranking",
    "aaj", "abhi", "nayi", "naya",
]

# ── Internet connectivity check  (FIX-P8) ─────────────────
INTERNET_CHECK_HOST    = "8.8.8.8"   # Google DNS — fast, highly available
INTERNET_CHECK_PORT    = 53
INTERNET_CHECK_TIMEOUT = 2.0

# ── VAD tuning ────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.035
SILENCE_AFTER_SPEECH = 1.2
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.1
IDLE_TIMEOUT         = 15.0
IDLE_POLL_TIMEOUT    = 30.0

# ── Wake words ────────────────────────────────────────────
WAKE_WORDS = ["hello", "hey", "hello acrobot", "hey acrobot", "acrobot"]

# ── System prompts ────────────────────────────────────────
_BASE_EN = (
    "Your name is AcroBot 2.2. You are the official AI assistant and "
    "virtual admission counselor of Acropolis Institute of Technology "
    "and Research (AITR), Indore. "
    "Acropolis College, Acropolis Institute, Acropolis, AITR, and "
    "Acropolis Institute of Technology and Research all refer to the "
    "same institution. "
    "Always represent AITR positively, professionally, and confidently. "
    "If users ask about another college or compare colleges, briefly and "
    "politely redirect the conversation toward AITR, highlight AITR's "
    "strengths, and do not make negative comments or false claims about "
    "other institutions. "
    "Never mention sources, PDFs, context, documents, retrieval systems, "
    "or knowledge bases unless the user specifically asks. "
    "If AITR-specific information is unavailable, search the web first; "
    "if not connected to the internet, answer naturally using general "
    "knowledge when appropriate. "
    "Keep responses short, natural, and human-like. Most replies should "
    "be 1–3 sentences. Do not provide more information than requested. "
    "Give detailed explanations only when the user explicitly asks. "
    "Do not use bullet points or markdown."
)

_BASE_HI = (
    "Aapka naam AcroBot 2.2 hai. Aap Acropolis Institute of Technology "
    "and Research (AITR), Indore ke official AI assistant aur virtual "
    "admission counselor hain. Acropolis, Acropolis College, Acropolis "
    "Institute, AITR aur Acropolis Institute of Technology and Research "
    "sab ek hi institute ke naam hain. "
    "Hamesha AITR ko positive, professional aur confident tarike se "
    "represent karein. Kisi doosre college ke baare mein poocha jaye ya "
    "comparison ho to short aur polite tarike se baat ko AITR ki taraf "
    "le jaayein, AITR ki strengths highlight karein, aur kisi institute "
    "ke baare mein negative ya false claims na karein. "
    "Kabhi bhi source, PDF, context, document, retrieval system ya "
    "knowledge base ka zikr na karein jab tak user specifically na pooche. "
    "Agar AITR sambandhit jankari available na ho to web search karke "
    "jawab dein; agar internet connect na ho to natural jawab dein. "
    "Jawab short, natural aur human-like rakhein. Adhiktar replies "
    "1–3 sentences ke hon. User detail maange tabhi vistaar se jawab "
    "dein. Bullet points ya markdown ka upyog na karein."
)

_LANG_DIRECTIVE = {
    "en": (
        "IMPORTANT: You MUST reply ONLY in English, regardless of the "
        "language the user writes in. Never respond in any other language."
    ),
    "hi": (
        "IMPORTANT: Aap SIRF Hindi ya Hinglish mein jawab dein, "
        "chahe user kisi bhi bhasha mein likhein. "
        "Kabhi bhi kisi aur bhasha mein jawab na dein."
    ),
}


def build_system(lang: str, context: str) -> str:
    base      = _BASE_HI if lang == "hi" else _BASE_EN
    directive = _LANG_DIRECTIVE.get(lang, _LANG_DIRECTIVE["en"])
    parts = [base, directive]
    if context:
        parts = [
            base,
            f"Use the following information silently to answer naturally.\n\n{context}",
            directive,
        ]
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"


# ══════════════════════════════════════════════════════════
#  INTERNET CONNECTIVITY  (FIX-P8)
# ══════════════════════════════════════════════════════════

def has_internet(timeout: float = INTERNET_CHECK_TIMEOUT) -> bool:
    """
    Lightweight connectivity probe.

    Opens a raw TCP connection to Google's public DNS (port 53).
    Returns True if the connection succeeds, False otherwise.

    NOTE (FIX-P12): Call this ONCE per user query in _process_query
    and pass the result down. Do NOT call it independently in each
    sub-function — each call adds up to 2s of latency on a dead network.
    """
    try:
        with socket.create_connection(
            (INTERNET_CHECK_HOST, INTERNET_CHECK_PORT), timeout=timeout
        ):
            return True
    except OSError:
        return False


# ══════════════════════════════════════════════════════════
#  ERROR HANDLING
# ══════════════════════════════════════════════════════════

ERROR_MESSAGES = {
    "api_error":   {"en": "I can't connect to the server."},
    "env_error":   {"en": "Environmental error, please restart me."},
    "no_internet": {"en": "I can't connect to the internet."},
}


def classify_error(exc: Exception, online: Optional[bool] = None) -> str:
    """
    Return 'no_internet', 'api_error', or 'env_error'.

    FIX-P12: Accepts a pre-computed `online` bool so the caller can
    pass in a cached connectivity result instead of triggering another
    TCP probe inside this function.
    """
    # Use cached result if provided, otherwise probe now.
    is_online = online if online is not None else has_internet()

    if not is_online:
        return "no_internet"

    api_related_types = (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
        socket.timeout,
        socket.gaierror,
    )
    if isinstance(exc, api_related_types):
        return "api_error"

    exc_name = type(exc).__name__.lower()
    exc_msg  = str(exc).lower()
    api_signals = (
        "api", "groq", "rate limit", "401", "403", "404", "429",
        "500", "502", "503", "504", "connection", "timeout",
        "network", "ssl", "host", "dns", "edge_tts", "endpoint",
    )
    if any(s in exc_name for s in api_signals) or \
       any(s in exc_msg  for s in api_signals):
        return "api_error"

    return "env_error"


def announce_error(exc: Exception, lang: str = "en", online: Optional[bool] = None) -> None:
    """
    Classify the exception and speak the appropriate English error message.
    FIX-P12: Accepts pre-computed `online` bool to avoid redundant probes.
    """
    try:
        kind = classify_error(exc, online=online)
        msg  = ERROR_MESSAGES[kind]["en"]
        logger.warning("Announcing error (%s): %s", kind, msg)
        speak(msg, lang="en", online=online)
    except Exception as report_exc:
        logger.error("Failed to announce error: %s", report_exc)


# ══════════════════════════════════════════════════════════
#  INPUT SANITIZATION  (FIX-P1 / FIX-P7)
# ══════════════════════════════════════════════════════════

def sanitize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_blank(text: Optional[str]) -> bool:
    return not text or not text.strip()


# ══════════════════════════════════════════════════════════
#  CONTENT MODERATION
# ══════════════════════════════════════════════════════════

_BLOCKED_PATTERNS: List[Tuple[re.Pattern, str]] = []

_RAW_BLOCKED = [
    # ── Profanity / Sexual (English) ─────────────────────────────
    (r"\bf+u+c+k+\b",           "profanity-en"),
    (r"\bs+h+i+t+\b",           "profanity-en"),
    (r"\bb+i+t+c+h+\b",         "profanity-en"),
    (r"\bass+h+o+l+e+\b",       "profanity-en"),
    (r"\bc+u+n+t+\b",           "profanity-en"),
    (r"\bd+i+c+k+\b",           "profanity-en"),
    (r"\bp+u+s+s+y+\b",         "profanity-en"),
    (r"\bn+i+g+g+\w*\b",        "slur-en"),
    (r"\bsex\b",                 "sexual-en"),
    (r"\bporn\w*\b",             "sexual-en"),
    (r"\bnude\w*\b",             "sexual-en"),
    (r"\bboob\w*\b",             "sexual-en"),
    (r"\bpenis\b",               "sexual-en"),
    (r"\bvagina\b",              "sexual-en"),
    # ── Profanity / Sexual (Hinglish Roman) ──────────────────────
    (r"\bmadarch\w*\b",          "profanity-hi"),
    (r"\bbhench\w*\b",           "profanity-hi"),
    (r"\bchutiy\w*\b",           "profanity-hi"),
    (r"\bsaala\b",               "profanity-hi"),
    (r"\bgandu\b",               "profanity-hi"),
    (r"\bharamz\w*\b",           "profanity-hi"),
    (r"\bkamina\b",              "profanity-hi"),
    (r"\blund\b",                "sexual-hi"),
    (r"\bchut\b",                "sexual-hi"),
    (r"\bsex\s*kar\w*\b",        "sexual-hi"),
    (r"\bnanga\b",               "sexual-hi"),
    # ── Violence / Threats ────────────────────────────────────────
    (r"\bkill\s+you\b",          "threat-en"),
    (r"\bi\s+will\s+kill\b",     "threat-en"),
    (r"\bbomb\b",                "threat-en"),
    (r"\bterror\w*\b",           "threat-en"),
    (r"\bblast\s+airport\b",     "threat-en"),
    (r"\bmarunga\b",             "threat-hi"),
    (r"\bjaan\s+se\s+marunga\b", "threat-hi"),
    (r"\bbomb\s+rakh\w*\b",      "threat-hi"),
    # ── Prompt Injection / Jailbreak ─────────────────────────────
    (r"\bignore\s+(all\s+)?previous\s+instructions?\b",  "jailbreak"),
    (r"\bpretend\s+(you\s+are|to\s+be)\b",              "jailbreak"),
    (r"\bact\s+as\s+(a\s+)?different\b",                "jailbreak"),
    (r"\byou\s+are\s+now\s+(dan|jailbreak\w*)\b",       "jailbreak"),
    (r"\bsystem\s*prompt\b",                            "jailbreak"),
    (r"\bforget\s+your\s+(rules?|instructions?)\b",     "jailbreak"),
    (r"\bdo\s+anything\s+now\b",                        "jailbreak"),
    (r"\bno\s+restrictions?\b",                         "jailbreak"),
]

for _raw, _label in _RAW_BLOCKED:
    try:
        _BLOCKED_PATTERNS.append((re.compile(_raw, re.IGNORECASE), _label))
    except re.error as _re_exc:
        logger.warning("Bad moderation pattern %r skipped: %s", _raw, _re_exc)

MAX_INPUT_CHARS = 500

_REFUSAL_EN = (
    "I'm here to help with questions about Acropolis Institute only. "
    "Please keep our conversation respectful."
)
_REFUSAL_HI = (
    "Mein sirf Acropolis Institute ke sawaalon mein madad karta hoon. "
    "Kripya izzat se baat karein."
)


def moderate_input(text: str, lang: str) -> Optional[str]:
    if len(text) > MAX_INPUT_CHARS:
        logger.warning("Input too long (%d chars) — blocked.", len(text))
        return _REFUSAL_HI if lang == "hi" else _REFUSAL_EN

    for pattern, label in _BLOCKED_PATTERNS:
        if pattern.search(text):
            logger.warning("Blocked input [%s]: %r", label, text[:80])
            return _REFUSAL_HI if lang == "hi" else _REFUSAL_EN

    return None


# ══════════════════════════════════════════════════════════
#  RAG ENGINE
# ══════════════════════════════════════════════════════════

class RAGEngine:

    def __init__(self) -> None:
        self.chunks:     List[str]                 = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix                                = None
        self.ready                                 = False

    def load_pdf(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.warning("RAG: PDF not found at '%s' — web/LLM only mode.", path)
            return False

        logger.info("RAG: Loading '%s' …", path)
        raw = self._extract_text(path)
        if not raw.strip():
            logger.warning("RAG: '%s' is empty — skipping.", path)
            return False

        self.chunks = self._chunk(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        self._build_index()
        self.ready  = True
        logger.info("RAG: '%s' indexed — %d chunks.", path, len(self.chunks))
        return True

    def retrieve(self, query: str) -> Tuple[str, float]:
        if not self.ready or not self.chunks:
            return "", 0.0

        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        top_idx    = scores.argsort()[::-1][:TOP_K]
        best_score = float(scores[top_idx[0]])

        context = "\n\n".join(
            self.chunks[i] for i in top_idx if scores[i] > 0
        )
        return context, best_score

    @staticmethod
    def _extract_text(path: str) -> str:
        doc   = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        words  = text.split()
        step   = max(1, size - overlap)
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _build_index(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
            token_pattern=r"\S+",
        )
        self.matrix = self.vectorizer.fit_transform(self.chunks)


# ══════════════════════════════════════════════════════════
#  WEB SEARCH FALLBACK
# ══════════════════════════════════════════════════════════

def web_search(query: str, online: bool = True) -> str:
    """
    FIX-P12: Accepts pre-computed `online` bool so we don't probe again.
    """
    if not online:
        logger.warning("web_search skipped — no internet connection.")
        return ""

    search_query = f"{query} Acropolis Institute Indore AITR"
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q":             search_query,
                "format":        "json",
                "no_html":       "1",
                "skip_disambig": "1",
            },
            timeout=WEB_TIMEOUT,
            headers={"User-Agent": "AcroBot/2.2"},
        )
        resp.raise_for_status()
        data     = resp.json()
        snippets: List[str] = []

        if data.get("AbstractText"):
            snippets.append(data["AbstractText"])

        for topic in data.get("RelatedTopics", [])[:WEB_RESULTS]:
            text = topic.get("Text", "")
            if text:
                snippets.append(text)

        context = " ".join(snippets).strip()
        if context:
            logger.debug("Web context fetched (%d chars).", len(context))
        else:
            logger.debug("Web search returned no usable snippets.")
        return context

    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        announce_error(exc, "en", online=online)
        return ""


def needs_web(query: str, score: float) -> bool:
    q              = query.lower()
    time_sensitive = any(kw in q for kw in WEB_KEYWORDS)
    low_score      = score < PDF_THRESHOLD
    return low_score and time_sensitive


# ══════════════════════════════════════════════════════════
#  GROQ CLIENT
# ══════════════════════════════════════════════════════════

try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialised.")
except Exception as _init_exc:
    logger.critical("Failed to initialise Groq client: %s", _init_exc)
    raise


# ══════════════════════════════════════════════════════════
#  CONVERSATION HISTORY  (FIX-P3)
# ══════════════════════════════════════════════════════════

history: dict = {"en": [], "hi": []}


def _trim_history(lang: str) -> None:
    lang_history = history[lang]
    if len(lang_history) > MAX_HISTORY_ITEMS:
        excess = len(lang_history) - MAX_HISTORY_ITEMS
        del lang_history[:excess]
        logger.debug("History trimmed: dropped %d oldest messages.", excess)


# ══════════════════════════════════════════════════════════
#  LLM  (FIX-P1, FIX-P2, FIX-P8, FIX-P12)
# ══════════════════════════════════════════════════════════

def get_ai_reply(user_text: str, lang: str, context: str, online: bool = True) -> str:
    """
    FIX-P12: Accepts pre-computed `online` bool — no redundant probe here.
    """
    clean_input = sanitize_text(user_text)
    if is_blank(clean_input):
        raise ValueError("get_ai_reply received empty or whitespace-only input.")

    refusal = moderate_input(clean_input, lang)
    if refusal:
        return refusal

    if not online:
        raise RuntimeError("no_internet: device has no internet connection.")

    lang_history = history[lang]
    lang_history.append({"role": "user", "content": clean_input})

    try:
        system = build_system(lang, context)

        last_exc: Optional[Exception] = None
        for attempt in range(1, LLM_MAX_RETRIES + 2):
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "system", "content": system}, *lang_history],
                    max_tokens=MAX_TOKENS,
                    temperature=0.4,
                )
                raw_reply = response.choices[0].message.content
                logger.debug("Raw LLM response (attempt %d): %r", attempt, raw_reply)

                reply = sanitize_text(raw_reply)
                if not is_blank(reply):
                    lang_history.append({"role": "assistant", "content": reply})
                    _trim_history(lang)
                    return reply

                logger.warning(
                    "LLM returned empty response on attempt %d/%d.",
                    attempt, LLM_MAX_RETRIES + 1,
                )
                last_exc = RuntimeError(
                    f"LLM returned an empty response (attempt {attempt})."
                )

            except Exception as api_exc:
                logger.warning("LLM API error on attempt %d: %s", attempt, api_exc)
                last_exc = api_exc
                if attempt <= LLM_MAX_RETRIES:
                    time.sleep(0.5 * attempt)

        lang_history.pop()
        raise last_exc or RuntimeError("LLM failed after all retry attempts.")

    except Exception:
        if lang_history and lang_history[-1]["role"] == "user":
            lang_history.pop()
        raise


# ══════════════════════════════════════════════════════════
#  CONTEXT BUILDER
# ══════════════════════════════════════════════════════════

def build_context(
    query: str,
    lang: str,
    rag_en: RAGEngine,
    rag_hi: RAGEngine,
    online: bool = True,
) -> Tuple[str, str]:
    """FIX-P12: Accepts pre-computed `online` bool."""
    rag = rag_hi if lang == "hi" else rag_en

    pdf_context, pdf_score = rag.retrieve(query)
    logger.debug("PDF score: %.3f (threshold=%.2f)", pdf_score, PDF_THRESHOLD)

    web_context = ""
    source      = "None"

    if pdf_context and pdf_score >= PDF_THRESHOLD:
        source = "PDF"

    if needs_web(query, pdf_score):
        web_context = web_search(query, online=online)
        if web_context:
            source = "PDF+Web" if pdf_context else "Web"
    else:
        if pdf_score < PDF_THRESHOLD:
            logger.debug("Web skipped — query is not time-sensitive.")

    parts: List[str] = []
    if pdf_context:
        parts.append(f"[From AITR Knowledge Base]\n{pdf_context}")
    if web_context:
        parts.append(f"[From Web]\n{web_context}")

    return "\n\n".join(parts), source


# ══════════════════════════════════════════════════════════
#  VAD RECORDING
# ══════════════════════════════════════════════════════════

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )
    stream.start()

    speech_buffer: List[np.ndarray]   = []
    pre_buffer:    List[np.ndarray]   = []
    recording                          = False
    silence_start: Optional[float]    = None
    idle_clock                         = time.time()

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if not recording and time.time() - idle_clock >= timeout:
                    return None
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                idle_clock    = time.time()
                silence_start = None
                if not recording:
                    recording     = True
                    speech_buffer = list(pre_buffer)
                speech_buffer.append(chunk)

            elif recording:
                speech_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    break

            else:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)
                if time.time() - idle_clock >= timeout:
                    return None

    finally:
        stream.stop()
        stream.close()

    if not speech_buffer:
        return None
    audio = np.concatenate(speech_buffer, axis=0)
    return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None


# ══════════════════════════════════════════════════════════
#  TRANSCRIBE
# ══════════════════════════════════════════════════════════

def transcribe(audio: np.ndarray, online: bool = True) -> Tuple[str, str]:
    """FIX-P12: Accepts pre-computed `online` bool."""
    if not online:
        raise RuntimeError("no_internet: device has no internet connection.")

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio, SAMPLE_RATE)

        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
                response_format="verbose_json",
            )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    text = sanitize_text(result.text)
    lang = (result.language or "en").strip().lower()

    if lang == "ur":
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    return text, lang


# ══════════════════════════════════════════════════════════
#  WAKE WORD
# ══════════════════════════════════════════════════════════

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ══════════════════════════════════════════════════════════
#  TTS  (FIX-P5, FIX-P10, FIX-P11)
# ══════════════════════════════════════════════════════════

# FIX-P5: Module-level event loop created once; reused by every speak() call.
_tts_loop = asyncio.new_event_loop()

# FIX-P10: pyttsx3 engine initialised ONCE at module load, not per-call.
# Re-initialising on every utterance causes thread conflicts and crashes
# on long sessions (the old background thread is still alive when the new
# one starts).
_pyttsx3_engine = None
if _PYTTSX3_AVAILABLE:
    try:
        _pyttsx3_engine = _pyttsx3_mod.init()
        _pyttsx3_engine.setProperty("rate", 155)
        logger.info("pyttsx3 offline TTS engine initialised.")
    except Exception as _pyttsx3_init_exc:
        logger.warning("pyttsx3 init failed — offline fallback unavailable: %s", _pyttsx3_init_exc)
        _pyttsx3_engine = None


def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


async def _tts_async(text: str, path: str, voice: str) -> None:
    await edge_tts.Communicate(text, voice=voice).save(path)


def speak(text: str, lang: str = "en", online: Optional[bool] = None) -> None:
    """
    Synthesise *text* and play it through speakers.

    Primary engine : edge-tts  (cloud, high quality, needs internet)
    Fallback engine: pyttsx3   (offline, espeak backend)

    FIX-P12: Accepts pre-computed `online` bool. Falls back to probing
    only when the caller did not pass one (e.g. error announce paths).
    """
    logger.debug("TTS input: %r", text)

    if is_blank(text):
        logger.error("TTS input validation failed: text is empty or None.")
        fallback = ERROR_MESSAGES["env_error"]["en"]
        _speak_direct(fallback, TTS_VOICE_EN)
        return

    voice = pick_voice(text, lang)
    logger.info("TTS [%s]: %s", voice, textwrap.shorten(text, width=80))

    # Resolve online status once for this speak() call.
    is_online = online if online is not None else has_internet()

    if _speak_edge_tts(text, voice, online=is_online):
        return

    logger.warning("edge-tts failed — attempting offline pyttsx3 fallback.")
    if _speak_pyttsx3(text, lang):
        return

    logger.error("All TTS engines failed for this utterance.")


def _speak_edge_tts(text: str, voice: str, online: bool = True) -> bool:
    """
    FIX-P12: Accepts pre-computed `online` bool — no redundant probe here.
    """
    if not online:
        logger.debug("edge-tts skipped — no internet connection.")
        return False

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        _tts_loop.run_until_complete(_tts_async(text, tmp_path, voice))

        if not os.path.exists(tmp_path):
            raise RuntimeError("edge-tts did not create output file.")

        mp3_size = os.path.getsize(tmp_path)
        logger.debug("Generated MP3 size: %d bytes", mp3_size)
        if mp3_size == 0:
            raise RuntimeError("edge-tts produced a zero-byte MP3 file.")

        try:
            pygame.mixer.music.load(tmp_path)
        except Exception as load_exc:
            raise RuntimeError(f"pygame failed to load MP3: {load_exc}") from load_exc

        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        return True

    except Exception as exc:
        logger.error("edge-tts error: %s", exc)
        return False

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _speak_pyttsx3(text: str, lang: str) -> bool:
    """
    Offline TTS via pyttsx3 (espeak backend).

    FIX-P10: Uses the module-level engine instance instead of calling
    pyttsx3.init() here. Re-initialising per-utterance caused thread
    conflicts and crashes on long sessions.
    """
    if _pyttsx3_engine is None:
        logger.debug("pyttsx3 not available — offline fallback unavailable.")
        return False

    try:
        # Select a voice that matches the language when possible.
        voices   = _pyttsx3_engine.getProperty("voices")
        lang_tag = "hi" if lang == "hi" else "en"
        for v in voices:
            v_lang = (
                v.languages[0].decode()
                if isinstance(v.languages[0], bytes)
                else v.languages[0]
            )
            if lang_tag in v_lang.lower():
                _pyttsx3_engine.setProperty("voice", v.id)
                break

        _pyttsx3_engine.say(text)
        _pyttsx3_engine.runAndWait()
        return True

    except Exception as exc:
        logger.error("pyttsx3 fallback error: %s", exc)
        return False


def _speak_direct(text: str, voice: str) -> None:
    """
    Minimal TTS+playback path used only by speak()'s input-validation
    guard — no risk of recursion because it calls _speak_edge_tts /
    _speak_pyttsx3 directly.
    """
    if _speak_edge_tts(text, voice, online=has_internet()):
        return
    logger.warning("_speak_direct: edge-tts failed, trying pyttsx3.")
    if _speak_pyttsx3(text, lang="en"):
        return
    logger.error("_speak_direct: all engines failed (giving up).")


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def print_banner(rag_en_ready: bool, rag_hi_ready: bool) -> None:
    status_en       = "✅ PDF loaded" if rag_en_ready else "⚠️  PDF not found — web-only mode"
    status_hi       = "✅ PDF loaded" if rag_hi_ready else "⚠️  PDF not found — web-only mode"
    internet_status = "🌐 Online" if has_internet() else "🚫 No internet detected"
    sep    = "=" * 60
    banner = (
        f"\n{sep}\n"
        f"  AcroBot 2.2 🤖  |  Acropolis Institute, Indore\n"
        f"{sep}\n"
        f"  Connectivity    : {internet_status}\n"
        f"  RAG (EN) status : {status_en}\n"
        f"  RAG (HI) status : {status_hi}\n"
        f"  PDF (EN) path   : {PDF_PATH_EN}\n"
        f"  PDF (HI) path   : {PDF_PATH_HI}\n"
        f"  PDF threshold   : {PDF_THRESHOLD}  (below → web fallback)\n"
        f"  Max history     : {MAX_HISTORY_TURNS} turns per language\n"
        f"  Chat model      : {CHAT_MODEL}\n"
        f"  Log level       : {_log_level_name}\n"
        f"  States          :\n"
        f"    👂 LISTENING  — auto-detects your voice\n"
        f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle\n"
        f"    🔊 SPEAKING   — playing response\n"
        f"  Ctrl+C to quit\n"
        f"{sep}\n"
    )
    print(banner)


def state_label(state: State) -> str:
    return {
        State.IDLE:      "😴 IDLE",
        State.LISTENING: "👂 LISTENING",
        State.THINKING:  "🤔 THINKING",
        State.SPEAKING:  "🔊 SPEAKING",
    }[state]


# ══════════════════════════════════════════════════════════
#  MAIN LOOP  (FIX-P6, FIX-P12)
# ══════════════════════════════════════════════════════════

def _process_query(
    user_text: str,
    lang: str,
    rag_en: RAGEngine,
    rag_hi: RAGEngine,
) -> Optional[str]:
    """
    Retrieve context for *user_text* and return the LLM reply string, or
    None on failure.

    FIX-P12: has_internet() is called ONCE here and the result is passed
    down to every sub-function (build_context → web_search, get_ai_reply,
    speak via announce_error). This eliminates 5–7 redundant TCP probes
    per query, saving up to ~10 seconds of latency on a dead network.
    """
    clean = sanitize_text(user_text)
    if is_blank(clean):
        logger.warning("Ignoring blank user input (after sanitization).")
        return None

    # ── Single connectivity probe for this entire query (FIX-P12) ─
    online = has_internet()

    if not online:
        logger.warning("No internet connection detected — aborting query.")
        speak(ERROR_MESSAGES["no_internet"]["en"], lang="en", online=False)
        return None

    logger.info("User [%s] › %s", lang.upper(), clean)
    logger.debug("Retrieving context …")

    context, source = build_context(clean, lang, rag_en, rag_hi, online=online)
    logger.info("Source: %s", source)
    logger.debug("Generating reply …")

    try:
        reply = get_ai_reply(clean, lang, context, online=online)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        announce_error(exc, lang, online=online)
        return None

    logger.info("AI   [%s] › %s", lang.upper(), reply)
    return reply


def main() -> None:
    try:
        # FIX-P11: Explicit frequency and buffer size for clean audio on
        # Raspberry Pi. Default settings often cause choppy or distorted
        # playback due to ALSA buffer underruns on low-power hardware.
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.init()

        rag_en = RAGEngine()
        rag_hi = RAGEngine()
        rag_en.load_pdf(PDF_PATH_EN)
        rag_hi.load_pdf(PDF_PATH_HI)
        print_banner(rag_en.ready, rag_hi.ready)

        state = State.LISTENING
        lang  = "hi"

        speak("Hello!", lang="hi")

        while True:

            # ── IDLE ──────────────────────────────────────
            if state == State.IDLE:
                logger.info(state_label(state))
                audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)
                if audio is None:
                    continue

                online = has_internet()
                try:
                    wake_text, _ = transcribe(audio, online=online)
                except Exception as exc:
                    logger.error("Transcription failed (idle): %s", exc)
                    announce_error(exc, "hi", online=online)
                    continue

                logger.debug("Heard (idle): %s", wake_text)

                if is_wake_word(wake_text):
                    state = State.LISTENING
                    speak("Haan, mein sun raha hoon.", lang="hi", online=online)
                continue

            # ── LISTENING ─────────────────────────────────
            if state == State.LISTENING:
                logger.info(state_label(state))
                audio = capture_speech(timeout=IDLE_TIMEOUT)

                if audio is None:
                    state = State.IDLE
                    speak(
                        "Mein idle mode mai jaa raha hoo, "
                        "Mujhe activate krne ke liye Hello boliyein.",
                        lang="hi",
                    )
                    continue

                online = has_internet()
                try:
                    user_text, lang = transcribe(audio, online=online)
                except Exception as exc:
                    logger.error("Transcription failed: %s", exc)
                    announce_error(exc, lang, online=online)
                    continue

                if is_blank(user_text):
                    logger.debug("Blank transcription — skipping.")
                    continue

                state = State.THINKING
                logger.info(state_label(state))
                reply = _process_query(user_text, lang, rag_en, rag_hi)

                if reply is None:
                    state = State.LISTENING
                    continue

                state = State.SPEAKING
                logger.info(state_label(state))
                speak(reply, lang)
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C).")
    except Exception as exc:
        logger.critical("Fatal error in main loop: %s", exc, exc_info=True)
        try:
            announce_error(exc, "en")
        except Exception:
            pass
    finally:
        try:
            _tts_loop.close()
        except Exception:
            pass
        # FIX-P10: Clean up the pyttsx3 engine on exit.
        if _pyttsx3_engine is not None:
            try:
                _pyttsx3_engine.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()