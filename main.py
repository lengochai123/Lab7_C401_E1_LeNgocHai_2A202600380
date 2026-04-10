from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# Configure stdout for UTF-8 to handle Vietnamese characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def get_sample_files(data_dir: str = "data") -> list[str]:
    """Dynamically load all .md and .txt files from specified directory.
    
    Prefers .md over .txt if both versions exist (e.g., python_intro.md vs python_intro.txt).
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        return []
    
    # Group files by stem (name without extension)
    files_by_stem: dict[str, list[Path]] = {}
    for file_path in dir_path.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".md", ".txt"}:
            stem = file_path.stem
            if stem not in files_by_stem:
                files_by_stem[stem] = []
            files_by_stem[stem].append(file_path)
    
    # Prefer .md over .txt if both exist
    result = []
    for stem in sorted(files_by_stem.keys()):
        candidates = files_by_stem[stem]
        # Sort to prefer .md
        candidates.sort(key=lambda p: (p.suffix.lower() != ".md", str(p)))
        result.append(str(candidates[0]))
    
    return result


SAMPLE_FILES = get_sample_files()


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    import re

    query = prompt.split("Question:", 1)[-1].strip()
    # Tìm pattern (YYYY–YYYY) hoặc (YYYY-YYYY) hoặc (YYYY — YYYY)
    range_match = re.search(r"\((1[0-9]{3}|20[0-9]{2})\s*[–-—-]\s*(1[0-9]{3}|20[0-9]{2})\)", prompt)
    q_lower = query.lower()
    if any(k in q_lower for k in ["sinh năm", "năm sinh", "sinh vào năm"]):
        if range_match:
            return f"[DEMO LLM] Sinh năm {range_match.group(1)}."
        year_matches = re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", prompt)
        if year_matches:
            return f"[DEMO LLM] Sinh năm {year_matches[0]}."
        return "[DEMO LLM] Không tìm thấy năm sinh trong ngữ cảnh."
    if any(k in q_lower for k in ["mất năm", "năm mất", "mất vào năm"]):
        if range_match:
            return f"[DEMO LLM] Mất năm {range_match.group(2)}."
        year_matches = re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", prompt)
        if len(year_matches) > 1:
            return f"[DEMO LLM] Mất năm {year_matches[1]}."
        elif year_matches:
            return f"[DEMO LLM] Mất năm {year_matches[0]}."
        return "[DEMO LLM] Không tìm thấy năm mất trong ngữ cảnh."

    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)

    # Áp dụng SentenceChunker: chunk mỗi document thành các đoạn nhỏ
    chunker = SentenceChunker(max_sentences_per_chunk=3)
    chunked_docs: list[Document] = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for i, chunk_text in enumerate(chunks):
            chunked_docs.append(
                Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=chunk_text,
                    metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": i},
                )
            )
    print(f"\nChunked {len(docs)} documents → {len(chunked_docs)} chunks (SentenceChunker, 3 sentences/chunk)")

    store.add_documents(chunked_docs)

    print(f"Stored {store.get_collection_size()} chunks in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    args = sys.argv[1:]
    
    # Check for --docs flag to load from data/docs
    use_docs_folder = "--docs" in args
    if use_docs_folder:
        args = [arg for arg in args if arg != "--docs"]
    
    question = " ".join(args).strip() if args else None
    
    # Determine which directory to use
    data_dir = "data/docs" if use_docs_folder else "data"
    sample_files = get_sample_files(data_dir)
    
    return run_manual_demo(question=question, sample_files=sample_files)


if __name__ == "__main__":
    raise SystemExit(main())
