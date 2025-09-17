# RAG Experiments - Code Repository Search

A comprehensive RAG (Retrieval-Augmented Generation) pipeline for searching through code repositories using semantic similarity and pattern matching.

## 🚀 Features

- **Semantic Code Search**: Find code using natural language queries
- **Multiple Search Modes**:
  - Chunks: Show best-matching code chunks
  - Function Definitions: Find function definitions
  - Function References: Find function calls
  - Variable References: Find variable usage
- **Web Interface**: Streamlit-based UI for easy interaction
- **Command Line Interface**: CLI for batch processing and automation
- **Multi-language Support**: Python, JavaScript, TypeScript, Java, Go, C++, and more

## 📁 Project Structure

```
RAG-Experiments/
├── db_creation.py          # Script to build FAISS index from code repository
├── db_retrieval.py         # CLI script for code retrieval
├── streamlit_retrieval.py  # Web interface for code search
├── requirements_streamlit.txt  # Dependencies for Streamlit app
├── faiss_store/            # Directory containing FAISS index and metadata (gitignored)
└── README.md
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd RAG-Experiments
```

2. **Install dependencies:**
```bash
pip install -r requirements_streamlit.txt
```

## 🚀 Quick Start

### 1. Build the FAISS Index

First, create a FAISS index from your code repository:

```bash
python db_creation.py /path/to/your/repo /path/to/faiss_store --ext .py .ts .tsx .js .jsx .md
```

**Options:**
- `--model`: SentenceTransformer model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--chunk-size`: Lines per chunk (default: 120)
- `--chunk-overlap`: Overlap between chunks (default: 20)
- `--ext`: File extensions to include

### 2. Search Using CLI

```bash
# Semantic search
python db_retrieval.py /path/to/faiss_store "how to create a trpc router" --mode chunks

# Find function definitions
python db_retrieval.py /path/to/faiss_store "find `AuthService` definitions" --mode function-defs

# Find function calls
python db_retrieval.py /path/to/faiss_store "where is `createUser` called" --mode function-refs

# Find variable references
python db_retrieval.py /path/to/faiss_store "DEBUG_MODE" --mode var-refs --identifier DEBUG_MODE
```

### 3. Search Using Web Interface

```bash
streamlit run streamlit_retrieval.py
```

Then open your browser to `http://localhost:8501`

## 📖 Usage Examples

### Semantic Search
- "how to implement authentication"
- "error handling patterns"
- "database connection setup"

### Function Search
- "find `AuthService` definitions"
- "where is `createUser` called"
- "`validateInput` function references"

### Variable Search
- "DEBUG_MODE usage"
- "API_KEY configuration"
- "database connection string"

## ⚙️ Configuration

### Supported File Extensions
- Python: `.py`
- JavaScript/TypeScript: `.js`, `.jsx`, `.ts`, `.tsx`
- Java: `.java`
- Go: `.go`
- C/C++: `.c`, `.cpp`, `.h`, `.hpp`
- Ruby: `.rb`
- PHP: `.php`
- And more...

### Search Modes
- **chunks**: Show best-matching code chunks (semantic search)
- **function-defs**: Find function definitions using regex patterns
- **function-refs**: Find function calls and references
- **var-refs**: Find variable usage and references

## 🔧 Advanced Usage

### Custom Models
You can use different SentenceTransformer models:

```bash
python db_creation.py /path/to/repo /path/to/faiss_store --model sentence-transformers/all-mpnet-base-v2
```

### Language Filtering
Filter results by programming language:

```bash
python db_retrieval.py /path/to/faiss_store "router" --mode function-refs --language typescript
```

### Batch Processing
Process multiple repositories:

```bash
for repo in repo1 repo2 repo3; do
    python db_creation.py /path/to/$repo /path/to/faiss_store_$repo
done
```

## 📊 Performance

- **Indexing**: ~1000 files per minute (depends on file size and model)
- **Search**: Sub-second response time for most queries
- **Memory**: ~500MB for 10k files (varies by model)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [SentenceTransformers](https://www.sbert.net/) for semantic embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Streamlit](https://streamlit.io/) for the web interface
