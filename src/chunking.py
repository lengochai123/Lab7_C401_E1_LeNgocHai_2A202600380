from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Split on sentence boundaries: ". ", "! ", "? ", or ".\n"
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + self.max_sentences_per_chunk])
            chunks.append(chunk)
        
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # Start recursive splitting with all separators
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """Recursively split text using remaining separators."""
        if not current_text:
            return []
        
        # If text fits within chunk_size, return it
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        # If no separators left, return the text as-is (oversized chunk)
        if not remaining_separators:
            return [current_text]
        
        # Try to split with the first separator
        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        
        if separator:
            # Split by this separator
            pieces = current_text.split(separator)
        else:
            # Empty separator means we split by character
            pieces = list(current_text)
        
        # Process each piece
        result = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                if piece:  # Only add non-empty pieces
                    result.append(piece)
            else:
                # Piece is too large, recurse with remaining separators
                sub_chunks = self._split(piece, next_separators)
                result.extend(sub_chunks)
        
        return result


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # Compute dot product
    dot_product = _dot(vec_a, vec_b)
    
    # Compute magnitudes
    mag_a = math.sqrt(sum(x * x for x in vec_a)) or 1.0
    mag_b = math.sqrt(sum(x * x for x in vec_b)) or 1.0
    
    # Guard against zero magnitude
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """
        Compare chunking strategies and return stats for each.
        
        Returns dict with keys 'fixed_size', 'by_sentences', 'recursive',
        each containing 'count', 'avg_length', and 'chunks'.
        """
        result = {}
        
        # FixedSizeChunker with chunk_size and no overlap
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=0)
        fixed_chunks = fixed_chunker.chunk(text)
        result['fixed_size'] = {
            'count': len(fixed_chunks),
            'avg_length': sum(len(c) for c in fixed_chunks) / len(fixed_chunks) if fixed_chunks else 0,
            'chunks': fixed_chunks
        }
        
        # SentenceChunker
        sent_chunker = SentenceChunker(max_sentences_per_chunk=3)
        sent_chunks = sent_chunker.chunk(text)
        result['by_sentences'] = {
            'count': len(sent_chunks),
            'avg_length': sum(len(c) for c in sent_chunks) / len(sent_chunks) if sent_chunks else 0,
            'chunks': sent_chunks
        }
        
        # RecursiveChunker
        rec_chunker = RecursiveChunker(chunk_size=chunk_size)
        rec_chunks = rec_chunker.chunk(text)
        result['recursive'] = {
            'count': len(rec_chunks),
            'avg_length': sum(len(c) for c in rec_chunks) / len(rec_chunks) if rec_chunks else 0,
            'chunks': rec_chunks
        }
        
        return result
