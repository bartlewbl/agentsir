"""Read and search files on this computer, and files sent in Telegram.

Workspace paths stay inside AGENT_FILES_ROOT, which is the directory the
process was started in unless that variable says otherwise. Uploads are kept
in memory for the conversation and read with the same tool.
"""

import os
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

from tools.registry import register
from tools.web.common import clean_text

_PART_CHARS = 6_000
_MAX_BYTES = 8 * 1024 * 1024
_MAX_TEXT = 200_000
_MAX_PDF_PAGES = 100
_MAX_UPLOADS = 8
_MAX_SCAN = 4_000
_MAX_HITS = 30
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
}
_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".xml", ".html", ".css", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".swift", ".kt", ".lua", ".sql", ".sh", ".r",
    ".tex",
}

_uploads: dict[str, dict] = {}
_by_user: dict[int, list[str]] = {}
_lock = threading.Lock()


def is_image(name: str, mime: str = "") -> bool:
    """True when a Telegram file should be looked at as a picture."""
    mime = mime.split(";")[0].strip().lower()
    return mime in _IMAGE_TYPES or Path(name).suffix.lower() in _IMAGE_SUFFIXES


def remember_upload(user_id: int, name: str, data: bytes) -> str:
    """Store an uploaded file and return the id `read_file` expects.

    Raises ValueError with a message for the user when the file can't be read.
    """
    text = _extract(name, data)
    safe_name = _safe_name(name)
    with _lock:
        file_id = _new_id()
        _uploads[file_id] = {"name": safe_name, "text": text, "user_id": user_id}
        order = _by_user.setdefault(user_id, [])
        order.append(file_id)
        while len(order) > _MAX_UPLOADS:
            _uploads.pop(order.pop(0), None)
    return file_id


def describe_upload(file_id: str, caption: str = "") -> str:
    """The user message that tells the agent to read and analyze an upload."""
    with _lock:
        item = _uploads.get(file_id)
    if item is None:
        return "That file is no longer available. Ask the user to send it again."
    parts = len(_split(item["text"]))
    ask = caption.strip() or "Read this and tell me what it says and what matters."
    return (
        f'The user sent a file "{item["name"]}" ({file_id}, {parts} part(s)). '
        "Read it with read_file before you answer. When there is more than one part, "
        "keep reading until you have what the question needs, then analyze it from "
        f"what you read.\n\n{ask}"
    )


def forget_uploads(user_id: int) -> None:
    """Drop the files a user sent, for example after /reset."""
    with _lock:
        for file_id in _by_user.pop(user_id, []):
            _uploads.pop(file_id, None)


@register
def list_files(path: str = ".") -> str:
    """List the files and folders in a workspace directory.

    path is relative to the workspace root unless it is already inside that
    root. Hidden tool directories are skipped. Use read_file on a file you find.
    """
    try:
        folder = _resolve(path)
    except PermissionError:
        return f"Could not list {path} (that path is outside the workspace)."
    if not folder.exists():
        return f"No folder at {path}."
    if not folder.is_dir():
        return f"{_label(folder)} is a file. Use read_file to open it."

    entries = []
    for child in sorted(folder.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if not _visible(child):
            continue
        entries.append(child)
    if not entries:
        return f"{_label(folder)} is empty."

    shown = entries[:200]
    lines = [_label(folder) + ":"]
    lines += [entry.name + ("/" if entry.is_dir() else "") for entry in shown]
    if len(entries) > len(shown):
        lines.append(f"({len(entries) - len(shown)} more not shown)")
    return "\n".join(lines)


@register
def search_files(query: str, path: str = ".") -> str:
    """Search workspace files for a word or phrase.

    Returns the file, the line number, and the matching line. Read the file
    to see what is around a hit. The match is exact aside from case.
    """
    query = " ".join(query.split())
    if not query:
        return "Give a word or phrase to search for."
    try:
        start = _resolve(path)
    except PermissionError:
        return f"Could not search {path} (that path is outside the workspace)."
    if not start.exists():
        return f"Nothing at {path}."

    if start.is_file() and not _visible(start):
        return f"Could not search {path} (that file looks like a secret)."
    needle = query.casefold()
    hits = []
    scanned = 0
    stopped = False
    files = [start] if start.is_file() else _walk(start)
    for file in files:
        scanned += 1
        if scanned > _MAX_SCAN:
            stopped = True
            break
        for hit in _search_file(file, needle):
            hits.append(hit)
            if len(hits) >= _MAX_HITS:
                stopped = True
                break
        if stopped:
            break

    where = _label(start)
    if not hits:
        return f'No matches for "{query}" in {where}.'
    noun = "match" if len(hits) == 1 else "matches"
    lines = [f'{len(hits)} {noun} for "{query}" in {where}:', ""]
    lines += hits
    if stopped:
        lines += ["", "Stopped early. Search a narrower folder if you need more."]
    return "\n".join(lines)


@register
def read_file(path: str, part: int = 1) -> str:
    """Read a workspace file or an uploaded file and return one part of its text.

    path is a workspace path, or an upload id such as upload:ab12cd34 from a
    file the user sent. Long files are split into parts. Ask for part=2, 3, ...
    to keep reading. Text, source code, and PDFs work. Read the relevant parts
    before analyzing a file.
    """
    upload_id = path.strip()
    if upload_id.startswith("upload:"):
        with _lock:
            item = _uploads.get(upload_id)
        if item is None:
            return f'No uploaded file "{upload_id}". Ask the user to send it again.'
        return _page(item["name"], item["text"], part)

    try:
        file = _resolve(path)
    except PermissionError:
        return f"Could not read {path} (that path is outside the workspace)."
    if _is_secret(file):
        return f"Could not read {path} (that file looks like a secret)."
    if not file.exists():
        return f"No file at {path}."
    if file.is_dir():
        return f"{_label(file)} is a directory. Use list_files to look inside."
    try:
        size = file.stat().st_size
    except OSError as exc:
        return f"Could not read {path} ({exc.__class__.__name__})."
    if size > _MAX_BYTES:
        return f"{_label(file)} is too big to read."
    try:
        text = _cached_text(str(file), file.stat().st_mtime_ns, size)
    except ValueError as exc:
        return str(exc)
    except OSError as exc:
        return f"Could not read {path} ({exc.__class__.__name__})."
    return _page(_label(file), text, part)


def _extract(name: str, data: bytes) -> str:
    if not data:
        raise ValueError("That file is empty.")
    if len(data) > _MAX_BYTES:
        raise ValueError("That file is too big to read. Send one under 8 MB.")
    if _is_pdf(name, data):
        text = _pdf_text(data)
    elif _looks_like_text(name, data):
        text = clean_text(_decode(data))
    else:
        raise ValueError(
            "I can read text files, source code, and PDFs. This one isn't one of those."
        )
    if not text:
        raise ValueError("That file has no readable text.")
    if len(text) > _MAX_TEXT:
        text = text[:_MAX_TEXT].rstrip() + f"\n\n[Stopped after {_MAX_TEXT} characters.]"
    return text


def _pdf_text(data: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    total = len(reader.pages)
    texts = [
        reader.pages[index].extract_text() or ""
        for index in range(min(total, _MAX_PDF_PAGES))
    ]
    text = clean_text("\n\n".join(texts))
    if total > _MAX_PDF_PAGES:
        text += f"\n\n[Stopped after {_MAX_PDF_PAGES} of {total} pages.]"
    return text


def _is_pdf(name: str, data: bytes) -> bool:
    return Path(name).suffix.lower() == ".pdf" or data.startswith(b"%PDF")


def _looks_like_text(name: str, data: bytes) -> bool:
    sample = data[:8192]
    if not sample or b"\x00" in sample:
        return False
    if Path(name).suffix.lower() in _TEXT_SUFFIXES:
        return True
    textish = sum(32 <= byte < 127 or byte in b"\t\n\r" for byte in sample)
    return textish / len(sample) > 0.9


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _page(label: str, text: str, part: int) -> str:
    parts = _split(text)
    total = len(parts)
    if isinstance(part, bool) or not isinstance(part, int) or not 1 <= part <= total:
        return f"{label} has {total} part(s)."
    body = parts[part - 1] or "(no readable text)"
    return f"File: {label}\nPart {part} of {total}\n\n{body}"


def _split(text: str) -> list[str]:
    parts = []
    current = ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > _PART_CHARS:
            parts.append(current)
            current = ""
        current += line
        while len(current) > _PART_CHARS:
            parts.append(current[:_PART_CHARS])
            current = current[_PART_CHARS:]
    if current.strip() or not parts:
        parts.append(current)
    return [part.strip() for part in parts]


def _root() -> Path:
    raw = os.environ.get("AGENT_FILES_ROOT", "").strip()
    root = Path(raw).expanduser() if raw else Path.cwd()
    return root.resolve()


def _resolve(path: str) -> Path:
    root = _root()
    raw = Path(path).expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not candidate.is_relative_to(root):
        raise PermissionError("outside the workspace")
    return candidate


def _label(path: Path) -> str:
    root = _root()
    try:
        relative = path.resolve().relative_to(root)
    except (OSError, ValueError):
        return path.name
    if relative == Path("."):
        return root.name
    return relative.as_posix()


def _visible(path: Path) -> bool:
    if path.name in _SKIP_DIRS or _is_secret(path):
        return False
    if path.is_dir() and path.name.startswith("."):
        return False
    try:
        return path.resolve().is_relative_to(_root())
    except OSError:
        return False


def _is_secret(path: Path) -> bool:
    name = path.name
    if name == ".env" or name.endswith((".pem", ".key", ".p12", ".pfx")):
        return True
    if name.startswith(".env.") and not name.endswith((".example", ".sample", ".template")):
        return True
    if name in {"id_rsa", "id_dsa", "id_ed25519", "id_ecdsa"}:
        return True
    return ".ssh" in path.parts


def _walk(folder: Path):
    files = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRS and not name.startswith("."))
        for name in sorted(filenames):
            file = Path(dirpath) / name
            if _visible(file) and file.is_file():
                files.append(file)
    return files


def _search_file(file: Path, needle: str) -> list[str]:
    try:
        if file.stat().st_size > _MAX_BYTES:
            return []
        data = file.read_bytes()
    except OSError:
        return []
    if not _looks_like_text(file.name, data[:8192]):
        return []
    text = _decode(data)
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line.casefold():
            snippet = " ".join(line.split())
            if len(snippet) > 200:
                snippet = snippet[:199].rstrip() + "…"
            found.append(f"{_label(file)}:{number}: {snippet}")
            if len(found) >= 8:
                break
    return found


@lru_cache(maxsize=32)
def _cached_text(path: str, mtime_ns: int, size: int) -> str:
    # mtime and size are part of the cache key, so a rewrite is read again.
    del mtime_ns, size
    return _extract(Path(path).name, Path(path).read_bytes())


def _safe_name(name: str) -> str:
    cleaned = Path(name or "file").name.replace("\x00", "").strip()
    return cleaned or "file"


def _new_id() -> str:
    while True:
        file_id = "upload:" + uuid.uuid4().hex[:8]
        if file_id not in _uploads:
            return file_id
