import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ChunkMetadata:
    repo_root: str
    relative_path: str
    start_line: int
    end_line: int
    language: Optional[str]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# -----------------------------
# File collection and filtering
# -----------------------------

DEFAULT_INCLUDE_EXTS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".yml", ".yaml",
    ".java", ".kt", ".go", ".rs", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".rb", ".php", ".swift", ".scala", ".sql", ".sh", ".bash", ".zsh",
    ".toml", ".ini", ".cfg", ".conf", ".cs"
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "node_modules", "dist", "build",
    "out", "coverage", "__pycache__", ".next", "venv", ".venv", "env"
}

DEFAULT_EXCLUDE_FILE_PATTERNS = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb", "poetry.lock",
    ".DS_Store"
}

BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".rar", ".7z", ".mp3", ".mp4", ".mov",
    ".wav", ".ogg", ".woff", ".woff2", ".ttf", ".eot"
}


def should_skip_path(path: Path, exclude_dirs: set[str]) -> bool:
    for part in path.parts:
        if part in exclude_dirs:
            return True
    return False


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTS:
        return False
    try:
        with open(path, "rb") as f:
            sample = f.read(2048)
        sample.decode("utf-8")
        return True
    except Exception:
        return False


# -----------------------------
# Chunking
# -----------------------------

LANGUAGE_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".java": "java",
    ".kt": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c-header",
    ".hpp": "cpp-header",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".md": "markdown",
}


def guess_language(path: Path) -> Optional[str]:
    return LANGUAGE_MAP.get(path.suffix.lower())


def chunk_lines(lines: List[str], chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    total = len(lines)
    while start < total:
        end = min(start + chunk_size, total)
        chunk_text = "".join(lines[start:end])
        # store 1-based line numbers for readability
        chunks.append((start + 1, end, chunk_text))
        if end == total:
            break
        start = end - chunk_overlap
    return chunks


# -----------------------------
# Embedding and indexing
# -----------------------------

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")


# -----------------------------
# Main build function
# -----------------------------

def build_faiss_index(
    repo_path: str,
    output_dir: str,
    include_exts: Optional[Iterable[str]] = None,
    exclude_dirs: Optional[Iterable[str]] = None,
    exclude_file_patterns: Optional[Iterable[str]] = None,
    max_file_size_bytes: int = 2_000_000,
    chunk_size: int = 120,
    chunk_overlap: int = 20,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> None:
    repo_root = Path(repo_path).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise FileNotFoundError(f"Repository path does not exist or is not a directory: {repo_root}")

    include_exts_set = set(include_exts) if include_exts else DEFAULT_INCLUDE_EXTS
    exclude_dirs_set = set(exclude_dirs) if exclude_dirs else DEFAULT_EXCLUDE_DIRS
    exclude_file_patterns_set = set(exclude_file_patterns) if exclude_file_patterns else DEFAULT_EXCLUDE_FILE_PATTERNS

    text_chunks: List[str] = []
    metadatas: List[ChunkMetadata] = []

    # Collect files
    candidates: List[Path] = []
    for path in repo_root.rglob("*"):
        if path.is_dir():
            if should_skip_path(path.relative_to(repo_root), exclude_dirs_set):
                continue
            else:
                continue
        # files
        if should_skip_path(path.relative_to(repo_root), exclude_dirs_set):
            continue
        if path.name in exclude_file_patterns_set:
            continue
        if path.suffix and include_exts_set and path.suffix.lower() not in include_exts_set:
            continue
        try:
            if path.stat().st_size > max_file_size_bytes:
                continue
        except OSError:
            continue
        if not is_text_file(path):
            continue
        candidates.append(path)

    if not candidates:
        raise RuntimeError("No source files found to index. Adjust include/exclude settings.")

    # Read and chunk
    for file_path in candidates:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = [line if line.endswith("\n") else line + "\n" for line in content.splitlines()]
        if not lines:
            continue
        for start_line, end_line, chunk_text in chunk_lines(lines, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            text_chunks.append(chunk_text)
            metadatas.append(
                ChunkMetadata(
                    repo_root=str(repo_root),
                    relative_path=str(file_path.relative_to(repo_root)),
                    start_line=start_line,
                    end_line=end_line,
                    language=guess_language(file_path),
                )
            )

    if not text_chunks:
        raise RuntimeError("No chunks produced from source files.")

    # Embed
    embedder = EmbeddingModel(model_name=model_name)
    embeddings = embedder.encode(text_chunks, batch_size=batch_size)

    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Prepare output dir
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save index
    faiss.write_index(index, str(out_dir / "index.faiss"))

    # Save metadata aligned with embeddings order
    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for meta in metadatas:
            f.write(meta.to_json() + "\n")

    # Save config
    config = {
        "model_name": model_name,
        "dim": embeddings.shape[1],
        "num_vectors": int(index.ntotal),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "repo_root": str(repo_root),
        "include_exts": sorted(list(include_exts_set)),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Built FAISS index with {index.ntotal} vectors at: {out_dir}")


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS DB for a code repository.")
    parser.add_argument("repo_path", type=str, help="Path to the repository root to index")
    parser.add_argument("output_dir", type=str, help="Directory to store index and metadata")
    parser.add_argument("--model", dest="model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model name")
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=120, help="Chunk size in lines")
    parser.add_argument("--chunk-overlap", dest="chunk_overlap", type=int, default=20, help="Chunk overlap in lines")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--max-file-size-bytes", dest="max_file_size_bytes", type=int, default=2_000_000,
                        help="Skip files larger than this size")
    parser.add_argument("--ext", dest="include_exts", type=str, nargs="*", default=None,
                        help="Override included extensions, e.g. --ext .py .ts .tsx")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    build_faiss_index(
        repo_path=args.repo_path,
        output_dir=args.output_dir,
        include_exts=args.include_exts,
        max_file_size_bytes=args.max_file_size_bytes,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
