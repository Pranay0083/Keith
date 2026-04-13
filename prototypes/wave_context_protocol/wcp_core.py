"""
Wave Context Protocol (WCP) — Core Engine
==========================================
Encodes text chunks as superposed sine waves and retrieves by FFT cross-correlation.
Two encoding modes:
  1. Neural:  chunk → all-MiniLM-L6-v2 → 384-dim → wave
  2. Native:  chunk → tokenize → hash-to-freq → wave  (no neural net, 100x faster)
"""

import numpy as np
import hashlib
import re
import os
import time
import json
import math
from collections import Counter
from typing import List, Tuple, Optional

# ─── Wave Parameters ────────────────────────────────────────────────────────
WAVE_SIZE = 4096          # sample points per wave
MAX_FREQ = 512            # number of unique frequency slots
BUCKET_CAPACITY = 2000    # max chunks per bucket before splitting
CHUNK_SIZE = 500          # chars per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks


# ─── Tokenizer ──────────────────────────────────────────────────────────────
# Splits code into meaningful tokens: identifiers, keywords, operators, literals
_TOKEN_RE = re.compile(r'[a-zA-Z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\s\w]')

def tokenize(text: str) -> List[str]:
    """Fast regex tokenizer tuned for source code."""
    return _TOKEN_RE.findall(text.lower())


# ─── TF-IDF Engine ──────────────────────────────────────────────────────────

class TFIDFEngine:
    """Lightweight TF-IDF computer. Builds IDF from ingested corpus."""

    def __init__(self):
        self.doc_freq = Counter()   # token → number of chunks containing it
        self.total_docs = 0

    def fit_chunk(self, tokens: List[str]):
        """Update IDF counts with a single chunk's unique tokens."""
        self.total_docs += 1
        unique = set(tokens)
        for t in unique:
            self.doc_freq[t] += 1

    def tfidf(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """Compute TF-IDF weights for a token list."""
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        result = []
        for token, count in tf.items():
            tf_val = count / total
            df = self.doc_freq.get(token, 0)
            idf = math.log((self.total_docs + 1) / (df + 1)) + 1.0
            result.append((token, tf_val * idf))
        return result


# ─── Hash-to-Frequency Mapping ──────────────────────────────────────────────

def token_to_freq(token: str) -> int:
    """Deterministically map a token to a frequency slot via hash."""
    h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
    return h % MAX_FREQ


# ─── Wave Encoding ──────────────────────────────────────────────────────────

# Pre-compute the time axis once (shared by all waves)
_T = np.linspace(0, 2 * np.pi, WAVE_SIZE, dtype=np.float32)

# Pre-compute sin lookup table for all frequencies: shape (MAX_FREQ, WAVE_SIZE)
_SIN_TABLE = np.array([np.sin(f * _T) for f in range(MAX_FREQ)], dtype=np.float32)


def encode_wave_native(text: str, tfidf_engine: TFIDFEngine) -> np.ndarray:
    """
    Native wave encoding — no neural net.
    Tokens are hashed to frequencies, weighted by TF-IDF, and summed into one wave.
    """
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(WAVE_SIZE, dtype=np.float32)

    weights = tfidf_engine.tfidf(tokens)
    wave = np.zeros(WAVE_SIZE, dtype=np.float32)

    for token, weight in weights:
        freq = token_to_freq(token)
        wave += weight * _SIN_TABLE[freq]

    # L2 normalize to unit energy
    norm = np.linalg.norm(wave)
    if norm > 0:
        wave /= norm

    return wave


def encode_wave_neural(embedding: np.ndarray) -> np.ndarray:
    """
    Neural wave encoding — converts a 384-dim embedding into a wave.
    Each dimension becomes the amplitude of a sine at a unique frequency.
    """
    dim = len(embedding)
    # Map 384 dims to evenly spaced frequencies across MAX_FREQ slots
    freqs = np.linspace(0, MAX_FREQ - 1, dim).astype(int)

    wave = np.zeros(WAVE_SIZE, dtype=np.float32)
    for i, freq in enumerate(freqs):
        wave += embedding[i] * _SIN_TABLE[freq]

    norm = np.linalg.norm(wave)
    if norm > 0:
        wave /= norm

    return wave


# ─── Bucket (Superposed Wave Container) ─────────────────────────────────────

class WaveBucket:
    """Holds a superposed wave of multiple chunks + metadata for drill-down."""

    def __init__(self, bucket_id: int):
        self.bucket_id = bucket_id
        self.superposed_wave = np.zeros(WAVE_SIZE, dtype=np.float32)
        self.chunk_waves: List[np.ndarray] = []
        self.chunk_meta: List[dict] = []  # {filepath, chunk_index, content}
        self.count = 0

    def add(self, wave: np.ndarray, filepath: str, chunk_index: int, content: str):
        self.superposed_wave += wave
        self.chunk_waves.append(wave)
        self.chunk_meta.append({
            'filepath': filepath,
            'chunk_index': chunk_index,
            'content': content,
        })
        self.count += 1

    @property
    def is_full(self) -> bool:
        return self.count >= BUCKET_CAPACITY


# ─── FFT Cross-Correlation (The Black Box Filter) ───────────────────────────

def wave_correlate(stored_wave: np.ndarray, query_wave: np.ndarray) -> float:
    """
    FFT cross-correlation between stored wave and query wave.
    Constructive interference → high peak → match.
    Destructive interference → low peak → no match.
    Returns the peak correlation amplitude.
    """
    # Matched filter in frequency domain
    F_stored = np.fft.rfft(stored_wave)
    F_query = np.fft.rfft(query_wave)

    # Multiply stored with conjugate of query → correlation
    correlation = np.fft.irfft(F_stored * np.conj(F_query), n=WAVE_SIZE)

    # Peak amplitude = resonance strength
    return float(np.max(np.abs(correlation)))


def wave_correlate_batch(stored_waves: List[np.ndarray], query_wave: np.ndarray) -> np.ndarray:
    """Vectorized FFT correlation against multiple waves at once."""
    if not stored_waves:
        return np.array([])

    stored_matrix = np.array(stored_waves)  # shape: (N, WAVE_SIZE)
    F_stored = np.fft.rfft(stored_matrix, axis=1)
    F_query = np.fft.rfft(query_wave)

    correlations = np.fft.irfft(F_stored * np.conj(F_query), n=WAVE_SIZE, axis=1)
    peaks = np.max(np.abs(correlations), axis=1)
    return peaks


# ─── WCP Index (The Full Engine) ────────────────────────────────────────────

class WCPIndex:
    """
    Wave Context Protocol Index.
    Ingests files into buckets of superposed waves.
    Searches via 2-level FFT cross-correlation.
    """

    SUPPORTED_EXTENSIONS = {
        '.c', '.h', '.go', '.py', '.js', '.ts', '.tsx', '.jsx',
        '.rs', '.cpp', '.cc', '.java', '.swift', '.zig', '.rb',
        '.md', '.txt', '.sh', '.json', '.html', '.css', '.yaml', '.yml',
        '.mod', '.sum', '.toml', '.cfg',
    }

    IGNORED_DIRS = {
        '.git', 'node_modules', 'venv', 'env', '__pycache__',
        'dist', 'build', 'vendor', '.vscode', '.idea', 'testdata',
        '.cache', '.tox', '.mypy_cache',
    }

    def __init__(self, mode: str = 'native'):
        """
        mode: 'native' (hash-to-freq, no ML) or 'neural' (all-MiniLM)
        """
        self.mode = mode
        self.buckets: List[WaveBucket] = [WaveBucket(0)]
        self.tfidf = TFIDFEngine()
        self.total_chunks = 0
        self._neural_model = None

    def _get_neural_model(self):
        if self._neural_model is None:
            from sentence_transformers import SentenceTransformer
            self._neural_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._neural_model

    def _current_bucket(self) -> WaveBucket:
        if self.buckets[-1].is_full:
            new_id = len(self.buckets)
            self.buckets.append(WaveBucket(new_id))
        return self.buckets[-1]

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        stride = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(text), stride):
            chunk = text[i:i + CHUNK_SIZE]
            if chunk.strip():
                chunks.append(chunk)
            if i + CHUNK_SIZE >= len(text):
                break
        return chunks

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to wave using the configured mode."""
        if self.mode == 'native':
            return encode_wave_native(text, self.tfidf)
        else:
            model = self._get_neural_model()
            emb = model.encode(text)
            return encode_wave_neural(emb)

    def _encode_batch_neural(self, texts: List[str]) -> List[np.ndarray]:
        """Batch encode for neural mode."""
        model = self._get_neural_model()
        embeddings = model.encode(texts, batch_size=512, show_progress_bar=False)
        return [encode_wave_neural(emb) for emb in embeddings]

    # ── Ingest ───────────────────────────────────────────────────────────────

    def ingest_file(self, filepath: str):
        """Ingest a single file into the wave index."""
        try:
            with open(filepath, 'r', errors='ignore') as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return 0

        if not content.strip():
            return 0

        chunks = self._chunk_text(content)

        # First pass: build TF-IDF stats (native mode only)
        if self.mode == 'native':
            for chunk in chunks:
                tokens = tokenize(chunk)
                self.tfidf.fit_chunk(tokens)

        # Second pass: encode and store
        if self.mode == 'neural' and len(chunks) > 1:
            waves = self._encode_batch_neural(chunks)
            for i, (chunk, wave) in enumerate(zip(chunks, waves)):
                bucket = self._current_bucket()
                bucket.add(wave, filepath, i, chunk)
                self.total_chunks += 1
        else:
            for i, chunk in enumerate(chunks):
                wave = self._encode(chunk)
                bucket = self._current_bucket()
                bucket.add(wave, filepath, i, chunk)
                self.total_chunks += 1

        return len(chunks)

    def ingest_directory(self, root_path: str, max_files: int = 0) -> Tuple[int, float]:
        """
        Walk a directory and ingest all supported files.
        Returns (total_chunks, elapsed_seconds).
        """
        root_path = os.path.abspath(root_path)
        files_to_ingest = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune ignored dirs in-place
            dirnames[:] = [d for d in dirnames if d not in self.IGNORED_DIRS]

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    files_to_ingest.append(os.path.join(dirpath, fname))

        if max_files > 0:
            files_to_ingest = files_to_ingest[:max_files]

        print(f"[WCP] Found {len(files_to_ingest)} files to ingest")

        # For native mode, do a 2-pass approach:
        # Pass 1: Build IDF stats from all chunks first
        if self.mode == 'native':
            print("[WCP] Pass 1/2: Building TF-IDF vocabulary...")
            t0 = time.time()
            all_file_chunks = []  # list of (filepath, list_of_chunks)

            for fp in files_to_ingest:
                try:
                    with open(fp, 'r', errors='ignore') as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if not content.strip():
                    continue
                chunks = self._chunk_text(content)
                all_file_chunks.append((fp, chunks))
                for chunk in chunks:
                    tokens = tokenize(chunk)
                    self.tfidf.fit_chunk(tokens)

            pass1_time = time.time() - t0
            print(f"[WCP] Pass 1 done: {self.tfidf.total_docs} chunks indexed for IDF in {pass1_time:.2f}s")

            # Pass 2: Encode and store
            print("[WCP] Pass 2/2: Encoding waves...")
            t0 = time.time()
            for fp, chunks in all_file_chunks:
                for i, chunk in enumerate(chunks):
                    wave = encode_wave_native(chunk, self.tfidf)
                    bucket = self._current_bucket()
                    bucket.add(wave, fp, i, chunk)
                    self.total_chunks += 1

            pass2_time = time.time() - t0
            total_time = pass1_time + pass2_time
            print(f"[WCP] Pass 2 done: {self.total_chunks} chunks encoded in {pass2_time:.2f}s")

        else:
            # Neural mode: single pass with batching
            t0 = time.time()
            batch_texts = []
            batch_meta = []  # (filepath, chunk_index)

            for fp in files_to_ingest:
                try:
                    with open(fp, 'r', errors='ignore') as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if not content.strip():
                    continue
                chunks = self._chunk_text(content)
                for i, chunk in enumerate(chunks):
                    batch_texts.append(chunk)
                    batch_meta.append((fp, i, chunk))

                    if len(batch_texts) >= 512:
                        self._ingest_neural_batch(batch_texts, batch_meta)
                        batch_texts = []
                        batch_meta = []

            # Flush remaining
            if batch_texts:
                self._ingest_neural_batch(batch_texts, batch_meta)

            total_time = time.time() - t0

        n_buckets = len(self.buckets)
        print(f"[WCP] Ingest complete: {self.total_chunks} chunks in {n_buckets} buckets ({total_time:.2f}s)")
        return self.total_chunks, total_time

    def _ingest_neural_batch(self, texts, metas):
        model = self._get_neural_model()
        embeddings = model.encode(texts, batch_size=512, show_progress_bar=False)
        for emb, (fp, idx, content) in zip(embeddings, metas):
            wave = encode_wave_neural(emb)
            bucket = self._current_bucket()
            bucket.add(wave, fp, idx, content)
            self.total_chunks += 1

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        2-level wave search:
        1. FFT correlate query wave against all bucket superposed waves → find top buckets
        2. FFT correlate query wave against individual chunks in winning buckets → rank results
        """
        t0 = time.time()

        # Encode query
        query_wave = self._encode(query)

        # Level 1: Score all buckets
        bucket_waves = [b.superposed_wave for b in self.buckets]
        bucket_scores = wave_correlate_batch(bucket_waves, query_wave)

        # Take top-3 buckets (or fewer if less exist)
        n_top_buckets = min(3, len(self.buckets))
        top_bucket_indices = np.argsort(bucket_scores)[-n_top_buckets:][::-1]

        # Level 2: Score individual chunks in top buckets
        all_results = []
        for bi in top_bucket_indices:
            bucket = self.buckets[bi]
            if not bucket.chunk_waves:
                continue

            chunk_scores = wave_correlate_batch(bucket.chunk_waves, query_wave)

            for j, score in enumerate(chunk_scores):
                meta = bucket.chunk_meta[j]
                all_results.append({
                    'filepath': meta['filepath'],
                    'chunk_index': meta['chunk_index'],
                    'content': meta['content'],
                    'score': float(score),
                    'bucket': bi,
                })

        # Sort by score descending
        all_results.sort(key=lambda x: x['score'], reverse=True)

        elapsed_ms = (time.time() - t0) * 1000
        top_results = all_results[:top_k]

        # Attach timing to results
        for r in top_results:
            r['query_time_ms'] = round(elapsed_ms, 3)

        return top_results

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        bucket_sizes = [b.count for b in self.buckets]
        total_wave_bytes = len(self.buckets) * WAVE_SIZE * 4  # float32 = 4 bytes
        total_chunk_wave_bytes = self.total_chunks * WAVE_SIZE * 4
        return {
            'mode': self.mode,
            'total_chunks': self.total_chunks,
            'num_buckets': len(self.buckets),
            'bucket_sizes': bucket_sizes,
            'bucket_wave_ram_kb': round(total_wave_bytes / 1024, 2),
            'chunk_wave_ram_mb': round(total_chunk_wave_bytes / (1024 * 1024), 2),
            'wave_size': WAVE_SIZE,
            'max_freq': MAX_FREQ,
            'bucket_capacity': BUCKET_CAPACITY,
        }


# ─── Convenience ─────────────────────────────────────────────────────────────

def create_index(mode: str = 'native') -> WCPIndex:
    """Factory function."""
    return WCPIndex(mode=mode)
