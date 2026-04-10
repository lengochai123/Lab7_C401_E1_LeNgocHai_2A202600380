from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        return {
            'id': doc.id,
            'content': doc.content,
            'embedding': self._embedding_fn(doc.content),
            'metadata': {
                'doc_id': doc.id,
                **doc.metadata
            }
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        if not records:
            return []
        
        # Embed the query
        query_embedding = self._embedding_fn(query)
        
        # Compute similarity scores
        scored_records = []
        for record in records:
            score = _dot(query_embedding, record['embedding'])
            scored_records.append({
                **record,
                'score': score
            })
        
        # Sort by score descending, take top_k
        scored_records.sort(key=lambda x: x['score'], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.
        Ưu tiên chunk thuộc đúng tài liệu chứa tên nhân vật trong query.
        """
        import unicodedata

        def _norm(s: str) -> str:
            return unicodedata.normalize("NFC", s).lower()

        # Tìm cụm tên riêng: chuỗi liên tiếp bắt đầu bằng chữ HOA
        query_nfc = unicodedata.normalize("NFC", query)
        words = query_nfc.split()
        name_phrases: list[str] = []
        current: list[str] = []
        for w in words:
            if w[0].isupper():
                current.append(w)
            else:
                if current:
                    name_phrases.append(" ".join(current))
                    current = []
        if current:
            name_phrases.append(" ".join(current))
        name_phrases = [p for p in name_phrases if len(p) >= 2]
        name_phrases.sort(key=len, reverse=True)

        if name_phrases:
            # Phân 3 nhóm: doc_id match > title line match > rest
            doc_id_match = []
            title_match = []
            rest = []
            for rec in self._store:
                doc_id = _norm(rec.get('metadata', {}).get('doc_id', ''))
                # Title = dòng đầu tiên của content (tên nhân vật)
                first_lines = _norm(rec['content'][:100].split('\n')[0])
                best = 0
                for phrase in name_phrases:
                    pn = _norm(phrase)
                    if pn in doc_id:
                        best = max(best, 3)
                    elif pn == first_lines or pn in first_lines:
                        best = max(best, 2)
                if best == 3:
                    doc_id_match.append(rec)
                elif best == 2:
                    title_match.append(rec)
                else:
                    rest.append(rec)

            # Ưu tiên: doc_id_match → title_match → rest
            # Trong mỗi nhóm, ưu tiên chunk_index nhỏ (header/tiểu sử) trước
            final: list[dict[str, Any]] = []
            for group in [doc_id_match, title_match, rest]:
                if group and len(final) < top_k:
                    # Sort: chunk_index=0 trước, rồi similarity
                    group_scored = self._search_records(query, group, len(group))
                    group_scored.sort(
                        key=lambda c: (
                            c.get('metadata', {}).get('chunk_index', 999),
                            -c.get('score', 0),
                        )
                    )
                    needed = top_k - len(final)
                    final.extend(group_scored[:needed])
            return final[:top_k]

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # If no filter, search all records
        if metadata_filter is None:
            return self.search(query, top_k)
        
        # Filter records by metadata
        filtered_records = []
        for record in self._store:
            # Check if all filter criteria match
            match = True
            for key, value in metadata_filter.items():
                if record['metadata'].get(key) != value:
                    match = False
                    break
            if match:
                filtered_records.append(record)
        
        # Search among filtered records
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        initial_size = len(self._store)
        self._store = [r for r in self._store if r['metadata'].get('doc_id') != doc_id]
        return len(self._store) < initial_size
