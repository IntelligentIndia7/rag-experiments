import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ChunkMeta:
    repo_root: str
    relative_path: str
    start_line: int
    end_line: int
    language: Optional[str]


# -----------------------------
# IO helpers
# -----------------------------

def load_config(index_dir: Path) -> dict:
    cfg_path = index_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {index_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(index_dir: Path) -> List[ChunkMeta]:
    meta_path = index_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl in {index_dir}")
    metas: List[ChunkMeta] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            metas.append(
                ChunkMeta(
                    repo_root=obj["repo_root"],
                    relative_path=obj["relative_path"],
                    start_line=int(obj["start_line"]),
                    end_line=int(obj["end_line"]),
                    language=obj.get("language"),
                )
            )
    return metas


def read_chunk_text(meta: ChunkMeta) -> Tuple[List[str], List[str]]:
    file_path = Path(meta.repo_root) / meta.relative_path
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        all_lines = f.readlines()
    # convert to 0-based indices for slicing
    start_idx = max(0, meta.start_line - 1)
    end_idx = min(len(all_lines), meta.end_line)
    return all_lines, all_lines[start_idx:end_idx]


# -----------------------------
# Embedding / Search
# -----------------------------

class QueryEncoder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32")


def search(index_dir: Path, query: str, top_k: int, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    index_path = index_dir / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.faiss in {index_dir}")
    index = faiss.read_index(str(index_path))
    encoder = QueryEncoder(model_name)
    q = encoder.encode(query)
    scores, ids = index.search(q, top_k)
    return scores[0], ids[0]


# -----------------------------
# Extraction utilities
# -----------------------------

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def guess_identifier_from_query(query: str) -> Optional[str]:
    backticked = re.findall(r"`([^`]+)`", query)
    if backticked:
        return backticked[-1].strip()
    # pick the last code-like token
    tokens = IDENTIFIER_PATTERN.findall(query)
    return tokens[-1] if tokens else None


# Language-aware regex patterns for function definitions
FUNC_DEF_PATTERNS = {
    "python": re.compile(r"^\s*def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("),
    "javascript": re.compile(r"(^|\s)(function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(|(?P<name2>[A-Za-z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>|(?P<name3>[A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{)"),
    "typescript": re.compile(r"(^|\s)(function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(|(?P<name2>[A-Za-z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>|(?P<name3>[A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{)"),
    "java": re.compile(r"^(\s*(public|private|protected)\s+)?(static\s+)?[\w<>\[\]]+\s+(?P<name>[A-Za-z_][\w_]*)\s*\("),
    "go": re.compile(r"^\s*func\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("),
    "cpp": re.compile(r"^(\s*(inline|static|virtual)\s+)?[\w:<>\*&\s]+\s+(?P<name>[A-Za-z_][\w_]*)\s*\("),
    "c": re.compile(r"^(\s*(static|inline|extern)\s+)?[\w\s\*]+\s+(?P<name>[A-Za-z_][\w_]*)\s*\("),
    "ruby": re.compile(r"^\s*def\s+(?P<name>[A-Za-z_][A-Za-z0-9_!?]*)\b"),
    "php": re.compile(r"^\s*function\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("),
}


def extract_function_defs(lines: List[str], language: Optional[str], target: Optional[str]) -> List[Tuple[int, str]]:
    results: List[Tuple[int, str]] = []
    pattern = FUNC_DEF_PATTERNS.get(language or "", None)
    for idx, line in enumerate(lines):
        name: Optional[str] = None
        if pattern:
            m = pattern.search(line)
            if m:
                name = m.group("name") or m.groupdict().get("name2") or m.groupdict().get("name3")
        else:
            m = re.search(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if m:
                name = m.group("name")
        if name:
            if target is None or name == target:
                results.append((idx, line.rstrip("\n")))
    return results


def extract_identifier_refs(lines: List[str], identifier: str, include_calls: bool, exclude_defs: bool) -> List[Tuple[int, str]]:
    results: List[Tuple[int, str]] = []
    escaped = re.escape(identifier)
    word = re.compile("\\b" + escaped + "\\b")
    call = re.compile("\\b" + escaped + "\\s*\\(")
    # Build a pattern that roughly matches function definitions for various languages
    def_pattern = (
        r"(def\s+" + escaped + r"\b|"  # Python/Ruby
        r"function\s+" + escaped + r"\b|"  # JS/PHP
        r"\b" + escaped + r"\s*\([^)]*\)\s*\{"  # JS/TS method or function style
        r")"
    )
    deflike = re.compile(def_pattern)
    for idx, line in enumerate(lines):
        if include_calls:
            if call.search(line):
                if exclude_defs and deflike.search(line):
                    continue
                results.append((idx, line.rstrip("\n")))
        else:
            if word.search(line):
                if exclude_defs and deflike.search(line):
                    continue
                results.append((idx, line.rstrip("\n")))
    return results


# -----------------------------
# Presentation
# -----------------------------

def format_match(repo_root: str, rel_path: str, chunk_start_line: int, local_idx: int, snippet: List[str], context: int = 2) -> str:
    start = max(0, local_idx - context)
    end = min(len(snippet), local_idx + context + 1)
    lines = []
    for i in range(start, end):
        lineno = chunk_start_line + i
        prefix = "=>" if i == local_idx else "  "
        lines.append(f"{prefix} {lineno:6d}: {snippet[i].rstrip('\n')}")
    header = f"{repo_root}/{rel_path}:{chunk_start_line + local_idx}"
    return header + "\n" + "\n".join(lines)


# -----------------------------
# Main query
# -----------------------------

def retrieve(
    index_dir: str,
    query: str,
    mode: str = "chunks",
    top_k: int = 10,
    identifier: Optional[str] = None,
    language_filter: Optional[str] = None,
) -> None:
    idx_dir = Path(index_dir).resolve()
    cfg = load_config(idx_dir)
    metas = load_metadata(idx_dir)

    scores, ids = search(idx_dir, query, top_k=top_k, model_name=cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))

    id_list = [i for i in ids if 0 <= i < len(metas)]

    print(f"Query: {query}")
    print(f"Mode: {mode}\n")

    target_identifier = identifier or guess_identifier_from_query(query)

    results_printed = 0
    for rank, idx in enumerate(id_list):
        meta = metas[idx]
        if language_filter and (meta.language or "").lower() != language_filter.lower():
            continue
        all_lines, chunk_lines = read_chunk_text(meta)
        if mode == "chunks":
            # Show the best-matching chunk
            highlight_idx = 0
            print(format_match(meta.repo_root, meta.relative_path, meta.start_line, highlight_idx, chunk_lines))
            print()
            results_printed += 1
        elif mode == "function-defs":
            defs = extract_function_defs(chunk_lines, meta.language, target_identifier)
            for local_idx, _ in defs:
                print(format_match(meta.repo_root, meta.relative_path, meta.start_line, local_idx, chunk_lines))
                print()
                results_printed += 1
        elif mode == "function-refs":
            if not target_identifier:
                continue
            refs = extract_identifier_refs(chunk_lines, target_identifier, include_calls=True, exclude_defs=True)
            for local_idx, _ in refs:
                print(format_match(meta.repo_root, meta.relative_path, meta.start_line, local_idx, chunk_lines))
                print()
                results_printed += 1
        elif mode == "var-refs":
            if not target_identifier:
                continue
            refs = extract_identifier_refs(chunk_lines, target_identifier, include_calls=False, exclude_defs=True)
            for local_idx, _ in refs:
                print(format_match(meta.repo_root, meta.relative_path, meta.start_line, local_idx, chunk_lines))
                print()
                results_printed += 1
        else:
            raise ValueError("Unsupported mode. Use one of: chunks, function-defs, function-refs, var-refs")

    if results_printed == 0:
        if target_identifier:
            print(f"No matches found for identifier '{target_identifier}' in retrieved chunks.")
        else:
            print("No matches found. Consider adjusting the query or mode.")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve code snippets from a FAISS DB for a repository.")
    parser.add_argument("index_dir", type=str, help="Directory containing index.faiss, metadata.jsonl, config.json")
    parser.add_argument("query", type=str, help="Natural-language or identifier query")
    parser.add_argument("--mode", type=str, default="chunks", choices=["chunks", "function-defs", "function-refs", "var-refs"],
                        help="Retrieval mode")
    parser.add_argument("--top-k", type=int, default=10, help="Number of nearest chunks to inspect")
    parser.add_argument("--identifier", type=str, default=None, help="Function/variable name to search for (overrides auto-detect)")
    parser.add_argument("--language", type=str, default=None, help="Filter by language label saved in metadata (e.g., python, typescript)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieve(
        index_dir=args.index_dir,
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
        identifier=args.identifier,
        language_filter=args.language,
    )


if __name__ == "__main__":
    main()
