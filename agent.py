"""
GAIA Agent — LangGraph + Gemini
================================
Grafo simple:  START -> assistant <-> tools -> END

- assistant : Gemini decide si responde o llama a una herramienta.
- tools     : ejecuta la herramienta y devuelve el resultado.
- Tope      : MAX_TOOL_CALLS llamadas por pregunta; al llegar se fuerza
              una respuesta final sin herramientas.
- Modelos   : lista de respaldo (GEMINI_MODELS). Ante 429/503 se espera y
              se pasa al siguiente modelo.

Mantiene la misma interfaz que usa app.py:
    GaiaAgent(question=...).run(task_id=..., question=..., file_name=...)
"""

import os
import re
import subprocess
import sys
import tempfile
import time

import requests

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Herramientas ya existentes (smolagents). Se envuelven más abajo.
from gaia_tools import (
    inspect_spreadsheet as _sm_inspect_spreadsheet,
    read_spreadsheet as _sm_read_spreadsheet,
    read_text_file as _sm_read_text_file,
    reverse_string as _sm_reverse_string,
    run_python_file as _sm_run_python_file,
    sum_spreadsheet_column as _sm_sum_spreadsheet_column,
    transcribe_audio as _sm_transcribe_audio,
)

try:
    from smolagents import DuckDuckGoSearchTool, VisitWebpageTool
    _sm_search = DuckDuckGoSearchTool(max_results=8)
    _sm_visit = VisitWebpageTool()
except Exception as _e:  # pragma: no cover
    print(f"[WARN] smolagents web tools unavailable: {_e}")
    _sm_search = None
    _sm_visit = None


# ============================================================
# CONFIGURATION
# ============================================================

# Lista de modelos de respaldo, separada por comas. El primero es el principal.
# Cuotas del nivel gratuito (AI Studio, oct 2026):
#   *-flash-lite : 15 RPM / 500 RPD   -> principales
#   *-flash      :  5 RPM /  20 RPD   -> respaldo
DEFAULT_MODELS = (
    "gemini-3.5-flash-lite,gemini-3.1-flash-lite,"
    "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,"
    "gemini-3.5-flash,gemini-3-flash,gemini-2.5-flash"
)
GEMINI_MODELS = [
    m.strip()
    for m in os.environ.get("GEMINI_MODELS", DEFAULT_MODELS).split(",")
    if m.strip()
]

MAX_TOOL_CALLS = int(os.environ.get("GAIA_MAX_TOOL_CALLS", "10"))
RETRY_WAIT_SECONDS = int(os.environ.get("GAIA_RETRY_WAIT", "20"))
MAX_SHORT_WAIT = 75          # si el 429 pide esperar <= 75 s, se espera y se repite el mismo modelo
MAX_MODEL_ATTEMPTS = int(os.environ.get("GAIA_MAX_MODEL_ATTEMPTS", "12"))
TOOL_OUTPUT_LIMIT = 12000  # caracteres que se devuelven al modelo por herramienta


# ============================================================
# SYSTEM PROMPTS (por tipo de pregunta)
# ============================================================

ANSWER_FORMAT_RULES = """
FINAL ANSWER FORMAT (GAIA is graded by EXACT match):
- Reply with ONLY the answer: no explanation, no prefix like "FINAL ANSWER:".
- A number: digits only, no commas, no units, no $ or % unless asked.
- A string: as few words as possible, no articles, no trailing period.
- A list: comma-separated, one space after each comma, alphabetical if asked.
- Never answer with a full sentence. Never say "I don't know": give your best answer.
"""

SYSTEM_PROMPT_GENERAL = """
You are an expert agent solving GAIA benchmark questions.
Use the tools when they make the answer reliable. Think step by step,
verify facts with tools, then reply with ONLY the final answer.

When a tool transforms or decodes information, read the full result: it may
contain a NEW INSTRUCTION. Execute that instruction and return its result,
not the intermediate text.
""" + ANSWER_FORMAT_RULES

SYSTEM_PROMPT_REVERSED_TEXT = """
You are solving a GAIA question with transformed or encoded text.
1. Use reverse_string (or the right transformation) on the text.
2. Read the result carefully: it usually contains a NEW INSTRUCTION.
3. Execute that instruction.
4. Reply with ONLY the final answer, never the intermediate text.
""" + ANSWER_FORMAT_RULES

SYSTEM_PROMPT_CODE = """
You are solving a GAIA question involving Python code.
1. Download the attachment with download_task_file(task_id).
2. Run it with run_python_file(file_path). If needed, read it with read_text_file.
3. Extract the exact value asked for from the output.
Never compute by hand what the code already computes.
""" + ANSWER_FORMAT_RULES

SYSTEM_PROMPT_DATA = """
You are solving a GAIA question involving spreadsheet data (Excel, CSV).
1. Download the attachment with download_task_file(task_id).
2. Inspect it with inspect_spreadsheet(file_path) to see columns and rows.
3. Compute with sum_spreadsheet_column or run_python_code (pandas).
NEVER estimate math mentally. Always compute with a tool.
""" + ANSWER_FORMAT_RULES

SYSTEM_PROMPT_SEARCH = """
You are solving a GAIA question that needs information lookup.
1. For anything on Wikipedia, use wikipedia_search then wikipedia_page(title, find=keyword).
   Wikipedia project pages (titles starting with 'Wikipedia:') hold featured-article logs,
   nomination pages, talk pages, etc. Read them with wikipedia_page too.
   If the question names a year/version of a page, pass as_of_date='YYYY-12-31'.
2. For other sources, use web_search and then visit_webpage on the best result.
   For a YouTube video, use youtube_transcript(url) to read what is said.
3. Count or compute with run_python_code when numbers are involved.
4. NEVER guess a name, number or date you did not read in a tool result. If a
   page is long, call wikipedia_page again with a different `find` keyword.
""" + ANSWER_FORMAT_RULES

SYSTEM_PROMPT_AUDIO = """
You are solving a GAIA question involving an audio attachment.
1. Download it with download_task_file(task_id).
2. Transcribe it with transcribe_audio(file_path).
3. Extract exactly what is requested (ingredients, page numbers, etc).
Return only the requested items. No measurements, no explanations.
""" + ANSWER_FORMAT_RULES


# ============================================================
# ROUTER: detect question type (same logic as before)
# ============================================================

def detect_question_type(question: str) -> str:
    q_lower = question.lower()

    if any(x in question for x in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]):
        return "audio_transcription"
    if any(x in q_lower for x in ["listen to", "voice memo", "recording", "audio file"]):
        return "audio_transcription"

    if question.startswith(".") and len(question) > 20:
        return "reversed_text"
    if question.count(".") > 3 and len(question) < 150:
        return "reversed_text"
    if any(x in q_lower for x in ["reverse", "reversed", "backwards", "upside", "understand this sentence"]):
        return "reversed_text"

    if any(x in question for x in [".py", "execute", "run the", "python file"]):
        return "python_code"
    if "python" in q_lower and ("code" in q_lower or "script" in q_lower):
        return "python_code"

    if any(x in question for x in [".xlsx", ".xls", ".csv"]):
        return "excel_data"
    if any(x in q_lower for x in ["spreadsheet", "column", "sum of", "total", "excel", "sheet", "table"]):
        return "excel_data"

    if "wikipedia" in q_lower:
        return "web_search"
    if any(x in q_lower for x in ["who", "when", "where", "what is", "biography", "founded", "born", "discovered", "created"]):
        return "web_search"
    if any(x in q_lower for x in ["author", "inventor", "scientist", "president", "studio albums"]):
        return "web_search"

    if any(x in q_lower for x in ["count", "occurs", "appears", "search in", "extract"]):
        return "text_extraction"
    if any(x in q_lower for x in ["if", "then", "depends", "compare", "which", "most", "least", "best", "worst"]):
        return "complex_logic"

    return "unknown"


def get_system_prompt(question_type: str) -> str:
    prompts = {
        "reversed_text": SYSTEM_PROMPT_REVERSED_TEXT,
        "python_code": SYSTEM_PROMPT_CODE,
        "excel_data": SYSTEM_PROMPT_DATA,
        "text_extraction": SYSTEM_PROMPT_DATA,
        "web_search": SYSTEM_PROMPT_SEARCH,
        "complex_logic": SYSTEM_PROMPT_SEARCH,
        "audio_transcription": SYSTEM_PROMPT_AUDIO,
        "unknown": SYSTEM_PROMPT_GENERAL,
    }
    return prompts.get(question_type, SYSTEM_PROMPT_GENERAL)


# ============================================================
# TOOLS (LangChain wrappers around the existing smolagents tools)
# ============================================================

def _clip(text) -> str:
    text = "" if text is None else str(text)
    if len(text) > TOOL_OUTPUT_LIMIT:
        return text[:TOOL_OUTPUT_LIMIT] + f"\n...[truncated, {len(text)} chars total]"
    return text


# ---------- Attachments: course API first, GAIA dataset on the Hub as fallback ----------

GAIA_API = "https://agents-course-unit4-scoring.hf.space"
GAIA_DATASET_FILES = "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/"
GAIA_DATASET_TREE = "https://huggingface.co/api/datasets/gaia-benchmark/GAIA/tree/main/2023/validation"
_CURRENT_TASK = {"task_id": None, "file_name": None}   # set by GaiaAgent.run()


def _hf_headers() -> dict:
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
           or os.environ.get("HF_API_TOKEN") or "")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _download_attachment(task_id: str, file_name: str | None = None) -> str:
    cache_dir = os.path.join(tempfile.gettempdir(), "gaia_files")
    os.makedirs(cache_dir, exist_ok=True)
    if not file_name and _CURRENT_TASK.get("task_id") == task_id:
        file_name = _CURRENT_TASK.get("file_name")
    if file_name:
        cached = os.path.join(cache_dir, os.path.basename(file_name))
        if os.path.exists(cached) and os.path.getsize(cached) > 0:
            print(f"[CACHE] Using cached file: {cached}")
            return cached

    errors = []

    # 1) Course API
    try:
        r = requests.get(f"{GAIA_API}/files/{task_id}", timeout=60)
        if r.ok and r.content:
            name = file_name
            if not name:
                cd = r.headers.get("content-disposition", "")
                if "filename=" in cd:
                    name = cd.split("filename=", 1)[1].strip().strip('"')
            name = os.path.basename(name or f"{task_id}.bin")
            path = os.path.join(cache_dir, name)
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"[DOWNLOAD] course API OK -> {path} ({len(r.content)} bytes)")
            return path
        errors.append(f"course API HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"course API: {e}")

    # 2) GAIA dataset on the Hub (gated: needs HF_TOKEN with access to gaia-benchmark/GAIA)
    try:
        if not file_name:
            t = requests.get(GAIA_DATASET_TREE, headers=_hf_headers(), timeout=60)
            if t.ok:
                for item in t.json():
                    base = os.path.basename(item.get("path", ""))
                    if base.startswith(task_id):
                        file_name = base
                        break
        if file_name:
            r = requests.get(GAIA_DATASET_FILES + file_name, headers=_hf_headers(), timeout=120)
            if r.ok and r.content:
                path = os.path.join(cache_dir, os.path.basename(file_name))
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"[DOWNLOAD] GAIA dataset OK -> {path} ({len(r.content)} bytes)")
                return path
            hint = " (set the HF_TOKEN secret with access to gaia-benchmark/GAIA)" if r.status_code in (401, 403) else ""
            errors.append(f"GAIA dataset HTTP {r.status_code}{hint}")
        else:
            errors.append("GAIA dataset: file name for this task not found")
    except Exception as e:
        errors.append(f"GAIA dataset: {e}")

    raise RuntimeError("; ".join(errors))


@tool
def download_task_file(task_id: str) -> str:
    """Download the attachment of a GAIA task. Returns the local file path to use with other tools."""
    try:
        return _download_attachment(task_id)
    except Exception as e:
        return f"ERROR: could not download the attachment: {e}"


@tool
def youtube_transcript(url: str) -> str:
    """Get the transcript (captions) of a YouTube video, with timestamps. Use it for questions about
    what is said in a video. It cannot see images; only spoken/caption text."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", url)
        vid = m.group(1) if m else url.strip()
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(vid, languages=["en", "en-US", "en-GB"])
            segs = [(s.start, s.text) for s in fetched]
        except AttributeError:  # older API versions
            data = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US", "en-GB"])
            segs = [(s["start"], s["text"]) for s in data]
        lines = [f"[{int(st // 60):02d}:{int(st % 60):02d}] {tx}" for st, tx in segs]
        return _clip("\n".join(lines) or "(empty transcript)")
    except Exception as e:
        return f"ERROR: transcript unavailable: {e}"


@tool
def read_text_file(file_path: str) -> str:
    """Read a text-based file (txt, py, json, md...) and return its contents."""
    return _clip(_sm_read_text_file(file_path=file_path))


@tool
def inspect_spreadsheet(file_path: str) -> str:
    """Show sheet names, columns, row counts and the first rows of an Excel or CSV file."""
    return _clip(_sm_inspect_spreadsheet(file_path=file_path))


@tool
def read_spreadsheet(file_path: str) -> str:
    """Return the full contents of an Excel or CSV file as text."""
    return _clip(_sm_read_spreadsheet(file_path=file_path))


@tool
def sum_spreadsheet_column(file_path: str, column_name: str) -> str:
    """Compute the exact sum of one column of an Excel or CSV file. column_name must match exactly."""
    return _clip(_sm_sum_spreadsheet_column(file_path=file_path, column_name=column_name))


@tool
def run_python_file(file_path: str) -> str:
    """Execute a local Python file and return its stdout/stderr."""
    return _clip(_sm_run_python_file(file_path=file_path))


@tool
def reverse_string(text: str) -> str:
    """Reverse a string character by character."""
    return _clip(_sm_reverse_string(text=text))


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file (mp3, wav, m4a...) to English text with Whisper."""
    return _clip(_sm_transcribe_audio(file_path=file_path))


@tool
def run_python_code(code: str) -> str:
    """Run a short Python snippet (pandas, numpy, re, math available) and return what it prints.
    Use print(...) to output the result. Timeout: 60 seconds."""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            path = f.name
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = result.stdout
        if result.stderr:
            out += "\nSTDERR:\n" + result.stderr
        return _clip(out.strip() or "(no output: use print)")
    except subprocess.TimeoutExpired:
        return "ERROR: Python execution timed out (60 seconds)"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo). Returns titles, links and snippets."""
    if _sm_search is None:
        return "ERROR: web search unavailable"
    try:
        return _clip(_sm_search(query=query))
    except Exception as e:
        return f"ERROR: search failed: {e}"


@tool
def visit_webpage(url: str) -> str:
    """Open a web page and return its text (markdown). Use it to read any URL from web_search."""
    if _sm_visit is None:
        return "ERROR: visit_webpage unavailable"
    try:
        return _clip(_sm_visit(url=url))
    except Exception as e:
        return f"ERROR: could not open page: {e}"


# ---------- Wikipedia (MediaWiki API) ----------

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "gaia-agent-eva/1.0 (HF Space; GAIA benchmark agent)"}


def _wiki_get(params: dict) -> dict:
    params = dict(params, format="json")
    r = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=40)
    r.raise_for_status()
    return r.json()


def _html_to_text(html: str) -> str:
    try:
        from markdownify import markdownify as md
        text = md(html, strip=["img", "script", "style"])
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\[edit\]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _focus(text: str, find: str, window: int = 1500, max_hits: int = 4) -> str:
    """Return only the passages around `find` (case-insensitive)."""
    if not find:
        return text
    hits = [m.start() for m in re.finditer(re.escape(find), text, re.I)]
    if not hits:
        return f"(keyword '{find}' not found; page has {len(text)} chars) Beginning of page:\n" + text[:3000]
    chunks, last_end = [], -1
    for h in hits:
        if len(chunks) >= max_hits:
            break
        if h < last_end:
            continue
        s, e = max(0, h - window), min(len(text), h + window)
        chunks.append(f"...[match {len(chunks) + 1} of {len(hits)} at char {h}]...\n{text[s:e]}")
        last_end = e
    return "\n\n".join(chunks)


@tool
def wikipedia_search(query: str) -> str:
    """Search English Wikipedia (articles AND project pages such as 'Wikipedia:Featured article candidates/...').
    Returns titles with snippets. Then read a page with wikipedia_page(title)."""
    try:
        j = _wiki_get({"action": "query", "list": "search", "srsearch": query,
                       "srlimit": 8, "srnamespace": "0|4"})
        rows = [
            f"- {r['title']}: {re.sub('<[^>]+>', '', r.get('snippet', ''))}"
            for r in j["query"]["search"]
        ]
        return "\n".join(rows) or "No results"
    except Exception as e:
        return f"ERROR: wikipedia_search failed: {e}"


@tool
def wikipedia_page(title: str, find: str = "", as_of_date: str = "") -> str:
    """Read an English Wikipedia page by its exact title, e.g. 'Mercedes Sosa' or a project page like
    'Wikipedia:Featured article candidates/Featured log/November 2016'. Tables are kept.
    find: optional keyword; if given, only the passages around it are returned (use it for long pages).
    as_of_date: optional 'YYYY-MM-DD'; returns the page as it was on that date (for 'latest 2022 version' questions)."""
    try:
        params = {"action": "parse", "prop": "text", "redirects": 1}
        if as_of_date:
            j = _wiki_get({"action": "query", "prop": "revisions", "titles": title, "redirects": 1,
                           "rvlimit": 1, "rvdir": "older", "rvstart": f"{as_of_date}T23:59:59Z",
                           "rvprop": "ids"})
            page = next(iter(j["query"]["pages"].values()))
            revs = page.get("revisions")
            if not revs:
                return f"ERROR: no revision of '{title}' exists before {as_of_date}"
            params["oldid"] = revs[0]["revid"]
        else:
            params["page"] = title
        j = _wiki_get(params)
        if "error" in j:
            return f"ERROR: {j['error'].get('info', j['error'])}"
        text = _html_to_text(j["parse"]["text"]["*"])
        return _clip(_focus(text, find))
    except Exception as e:
        return f"ERROR: wikipedia_page failed: {e}"


ALL_TOOLS = [
    wikipedia_search,
    wikipedia_page,
    web_search,
    visit_webpage,
    youtube_transcript,
    download_task_file,
    read_text_file,
    inspect_spreadsheet,
    read_spreadsheet,
    sum_spreadsheet_column,
    run_python_file,
    run_python_code,
    reverse_string,
    transcribe_audio,
]


# ============================================================
# MODEL with fallback list
# ============================================================

_RETRYABLE = ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded",
              "rate limit", "high demand", "Too Many Requests", "quota")
_MODEL_GONE = ("404", "NOT_FOUND", "not found", "is not supported", "not available")


def _is_retryable(err: Exception) -> bool:
    msg = str(err)
    return any(k.lower() in msg.lower() for k in _RETRYABLE)


def _is_model_gone(err: Exception) -> bool:
    msg = str(err)
    return any(k.lower() in msg.lower() for k in _MODEL_GONE)


def _retry_after_seconds(err: Exception):
    """Lee 'retry in 20h5m50s' / 'retryDelay': '12s' del error de Gemini. None si no aparece."""
    msg = str(err)
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", msg, re.I)
    if m:
        h, mi, s = m.groups()
        return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s)
    m = re.search(r"retryDelay['\"]?:\s*['\"]?([\d.]+)s", msg)
    if m:
        return float(m.group(1))
    return None


def _make_llm(model_name: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.environ["GEMINI_API_KEY"],
        max_retries=1,
        timeout=90,
    )


class FallbackLLM:
    """
    Invoke Gemini with a pool of models.
    - 429 por minuto (retry corto)  -> espera y repite el MISMO modelo.
    - 429 por día / 503             -> pasa al SIGUIENTE modelo de la lista.
    - 404 (modelo inexistente)      -> saca ese modelo de la lista.
    """

    def __init__(self, models, tools):
        self.models = list(models)
        self.tools = tools
        self.index = 0

    def _llm(self, with_tools: bool):
        llm = _make_llm(self.models[self.index])
        return llm.bind_tools(self.tools) if with_tools else llm

    def _next_model(self):
        self.index = (self.index + 1) % len(self.models)

    def invoke(self, messages, with_tools: bool = True) -> AIMessage:
        last_err = None
        for attempt in range(MAX_MODEL_ATTEMPTS):
            if not self.models:
                break
            name = self.models[self.index]
            try:
                return self._llm(with_tools).invoke(messages)
            except Exception as e:
                last_err = e
                print(f"[LLM] {name} failed (attempt {attempt + 1}): {str(e)[:160]}")

                if _is_model_gone(e) and not _is_retryable(e):
                    print(f"[LLM] model '{name}' not available: removing it from the pool")
                    self.models.pop(self.index)
                    if not self.models:
                        break
                    self.index %= len(self.models)
                    continue

                if not _is_retryable(e):
                    raise

                wait = _retry_after_seconds(e)
                if wait is not None and wait <= MAX_SHORT_WAIT:
                    # Límite por minuto: esperar y repetir con el mismo modelo
                    print(f"[LLM] per-minute limit, waiting {wait:.0f}s on {name}")
                    time.sleep(wait + 1)
                    continue

                # Cuota diaria agotada o modelo saturado: cambiar de modelo
                self._next_model()
                pause = RETRY_WAIT_SECONDS if self.index == 0 else 2
                print(f"[LLM] switching to {self.models[self.index]} (pause {pause}s)")
                time.sleep(pause)
        raise RuntimeError(f"All model attempts failed: {last_err}")


# ============================================================
# GRAPH
# ============================================================

def build_graph(llm: FallbackLLM, system_prompt: str):

    def assistant(state: MessagesState):
        messages = state["messages"]
        tool_calls_done = sum(isinstance(m, ToolMessage) for m in messages)
        msgs = [SystemMessage(content=system_prompt)] + list(messages)

        if tool_calls_done >= MAX_TOOL_CALLS:
            print(f"[GRAPH] Tool budget reached ({tool_calls_done}). Forcing final answer.")
            msgs.append(HumanMessage(content=(
                "You have no more tool calls available. Using everything above, "
                "reply now with ONLY your best final answer in the required format."
            )))
            return {"messages": [llm.invoke(msgs, with_tools=False)]}

        return {"messages": [llm.invoke(msgs, with_tools=True)]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile()


# ============================================================
# GAIA AGENT (same interface as before)
# ============================================================

class GaiaAgent:

    def __init__(self, question: str = None):
        self.question_type = detect_question_type(question) if question else "unknown"
        self.system_prompt_text = get_system_prompt(self.question_type)

        print(f"\n[ROUTER] Question type: {self.question_type}")
        print(f"[LLM] Models: {GEMINI_MODELS}  | max tool calls: {MAX_TOOL_CALLS}")

        if "GEMINI_API_KEY" not in os.environ:
            raise RuntimeError("GEMINI_API_KEY is not set in the Space secrets")

        self.llm = FallbackLLM(GEMINI_MODELS, ALL_TOOLS)
        self.graph = build_graph(self.llm, self.system_prompt_text)
        print("✓ GAIA agent (LangGraph) ready\n")

    # --------------------------------------------------------
    def run(self, task_id: str, question: str, file_name: str | None = None):
        print("\n" + "=" * 60)
        print("GAIA QUESTION")
        print("=" * 60)
        print(question)
        if file_name:
            print(f"\nAttachment: {file_name}")
        print("=" * 60 + "\n")

        _CURRENT_TASK["task_id"] = task_id
        _CURRENT_TASK["file_name"] = file_name or None

        user_prompt = f"GAIA TASK ID: {task_id}\n"
        if file_name:
            user_prompt += (
                f"ATTACHMENT: {file_name} "
                f"(download it with download_task_file(\"{task_id}\"))\n"
            )
        user_prompt += f"\nQUESTION:\n{question}"

        try:
            recursion_limit = 2 * MAX_TOOL_CALLS + 6
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=user_prompt)]},
                config={"recursion_limit": recursion_limit},
            )
            final = self._last_ai_text(result["messages"])
            answer = self.clean_answer(final)

            if not answer:
                print("[GRAPH] Empty answer, asking once more without tools.")
                msgs = self._trim_dangling(result["messages"])
                msgs = [SystemMessage(content=self.system_prompt_text)] + msgs + [
                    HumanMessage(content="Reply now with ONLY the final answer.")
                ]
                answer = self.clean_answer(self.llm.invoke(msgs, with_tools=False).content)

            print("\n" + "=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)
            print(answer)
            print("=" * 60 + "\n")
            return answer

        except Exception as e:
            error_msg = f"Agent error: {repr(e)}"
            print(f"✗ {error_msg}")
            return f"ERROR: {error_msg}"

    # --------------------------------------------------------
    @staticmethod
    def _last_ai_text(messages) -> str:
        for m in reversed(messages):
            if isinstance(m, AIMessage) and not m.tool_calls:
                return GaiaAgent._content_to_text(m.content)
        return ""

    @staticmethod
    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _trim_dangling(messages):
        """Drop a trailing AIMessage with unanswered tool calls (Gemini rejects it)."""
        msgs = list(messages)
        while msgs and isinstance(msgs[-1], AIMessage) and msgs[-1].tool_calls:
            msgs.pop()
        return msgs

    # --------------------------------------------------------
    def clean_answer(self, result) -> str:
        if result is None:
            return ""
        answer = self._content_to_text(result).strip()

        prefixes = [
            "FINAL ANSWER:", "Final Answer:", "final answer:",
            "ANSWER:", "Answer:", "The answer is:", "The answer:",
            "Based on my analysis, the answer is:",
        ]
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
                    changed = True

        # Si hay varias líneas, quedarse con la última corta (suele ser la respuesta)
        if "\n" in answer:
            lines = [l.strip() for l in answer.split("\n") if l.strip()]
            short = [l for l in lines if len(l) < 100]
            answer = short[-1] if short else " ".join(answer.split())

        answer = re.sub(r"\*\*", "", answer).strip().strip("`").strip()
        for _ in range(2):
            if answer.endswith("."):
                answer = answer[:-1].strip()
            if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in "\"'":
                answer = answer[1:-1].strip()
        return " ".join(answer.split())
