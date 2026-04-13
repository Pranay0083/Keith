#!/usr/bin/env python3
"""
WCP Benchmark Suite
====================
Tests the Wave Context Protocol against Keith's own codebase and the Linux kernel.
Measures: ingest time, query time, precision, RAM usage.
"""

import sys
import os
import time
import tracemalloc
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from wcp_core import WCPIndex

# ─── Config ──────────────────────────────────────────────────────────────────

KEITH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LINUX_ROOT = os.path.abspath(os.path.join(KEITH_ROOT, '..', 'linux-kernel-core-test'))

# Ground truth queries for Keith codebase
KEITH_QUERIES = [
    {
        'query': 'vector embedding search cosine similarity',
        'expect_files': ['memory.go', 'embedder.go'],
        'description': 'Find vector search implementation',
    },
    {
        'query': 'websocket server connection upgrade',
        'expect_files': ['websocket.go', 'server.go'],
        'description': 'Find websocket handling',
    },
    {
        'query': 'agent loop tool execution LLM',
        'expect_files': ['loop.go', 'tools.go'],
        'description': 'Find the agent loop',
    },
    {
        'query': 'oauth token authentication login',
        'expect_files': ['oauth.go', 'token.go'],
        'description': 'Find auth/login code',
    },
    {
        'query': 'telegram bot message send',
        'expect_files': ['telegram.go'],
        'description': 'Find Telegram integration',
    },
    {
        'query': 'ingest directory recursive walk file',
        'expect_files': ['rag.go'],
        'description': 'Find file ingestion code',
    },
    {
        'query': 'system brightness volume hardware control',
        'expect_files': ['tools_system.go'],
        'description': 'Find hardware control tools',
    },
    {
        'query': 'self healing watchdog restart crash recovery',
        'expect_files': ['tools_selfheal.go', 'watchdog.go'],
        'description': 'Find self-healing system',
    },
    {
        'query': 'calendar event schedule meeting create',
        'expect_files': ['tools_calendar.go', 'tools_google_calendar.go'],
        'description': 'Find calendar tools',
    },
    {
        'query': 'gemini API stream generate content response',
        'expect_files': ['gemini.go', 'client.go'],
        'description': 'Find Gemini API client',
    },
]

# Ground truth queries for Linux kernel
LINUX_QUERIES = [
    {
        'query': 'memory allocation kmalloc slab page',
        'expect_files': ['kmalloc', 'slab', 'page_alloc'],
        'description': 'Find memory allocator',
    },
    {
        'query': 'scheduler task switch context process',
        'expect_files': ['sched', 'context_switch', 'core.c'],
        'description': 'Find scheduler code',
    },
    {
        'query': 'mutex lock unlock spinlock synchronization',
        'expect_files': ['mutex', 'spinlock', 'lock'],
        'description': 'Find locking primitives',
    },
    {
        'query': 'network socket TCP connect send receive',
        'expect_files': ['tcp', 'socket', 'sock'],
        'description': 'Find TCP networking',
    },
    {
        'query': 'filesystem inode dentry open read write',
        'expect_files': ['inode', 'dentry', 'namei', 'open.c'],
        'description': 'Find VFS layer',
    },
    {
        'query': 'interrupt handler IRQ request register',
        'expect_files': ['irq', 'interrupt', 'handler'],
        'description': 'Find interrupt handling',
    },
    {
        'query': 'USB device driver probe endpoint transfer',
        'expect_files': ['usb', 'driver', 'probe'],
        'description': 'Find USB subsystem',
    },
    {
        'query': 'ext4 journal block bitmap superblock',
        'expect_files': ['ext4', 'journal', 'super.c'],
        'description': 'Find ext4 filesystem',
    },
]


def calc_precision(results: list, expected_patterns: list, k: int = 5) -> float:
    """
    Precision@K: what fraction of top-K results contain an expected filename pattern.
    """
    top_k = results[:k]
    hits = 0
    for r in top_k:
        fp = r['filepath'].lower()
        for pattern in expected_patterns:
            if pattern.lower() in fp:
                hits += 1
                break
    return hits / k if k > 0 else 0.0


def run_benchmark(name: str, root_path: str, queries: list, mode: str = 'native',
                  max_files: int = 0):
    """Run a full benchmark on a codebase."""
    print(f"\n{'='*80}")
    print(f"  BENCHMARK: {name} ({mode} mode)")
    print(f"  Path: {root_path}")
    print(f"{'='*80}\n")

    if not os.path.isdir(root_path):
        print(f"  SKIP — directory not found: {root_path}")
        return None

    # Track memory
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]

    # Create index and ingest
    index = WCPIndex(mode=mode)
    total_chunks, ingest_time = index.ingest_directory(root_path, max_files=max_files)

    mem_after = tracemalloc.get_traced_memory()[0]
    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    stats = index.stats()

    print(f"\n--- Ingest Results ---")
    print(f"  Chunks:          {total_chunks:,}")
    print(f"  Buckets:         {stats['num_buckets']}")
    print(f"  Ingest time:     {ingest_time:.2f}s")
    print(f"  Chunks/sec:      {total_chunks / ingest_time:,.0f}" if ingest_time > 0 else "  N/A")
    print(f"  Bucket RAM:      {stats['bucket_wave_ram_kb']:.1f} KB")
    print(f"  Chunk waves RAM: {stats['chunk_wave_ram_mb']:.1f} MB")
    print(f"  Actual RAM delta: {(mem_after - mem_before) / (1024*1024):.1f} MB")
    print(f"  Peak RAM:        {peak_mem / (1024*1024):.1f} MB")

    # Run queries
    print(f"\n--- Query Results ---")
    total_precision = 0.0
    total_query_time = 0.0
    query_results = []

    for q in queries:
        results = index.search(q['query'], top_k=10)
        query_time = results[0]['query_time_ms'] if results else 0

        precision = calc_precision(results, q['expect_files'], k=5)
        total_precision += precision
        total_query_time += query_time

        status = "✓" if precision >= 0.4 else "✗"
        print(f"  {status} [{precision:.0%} P@5, {query_time:.1f}ms] {q['description']}")

        if results:
            # Show top 3 results
            for r in results[:3]:
                fname = os.path.basename(r['filepath'])
                print(f"      → {fname} (chunk {r['chunk_index']}, score={r['score']:.4f})")

        query_results.append({
            'query': q['query'],
            'description': q['description'],
            'precision_at_5': precision,
            'query_time_ms': query_time,
            'top_3_files': [os.path.basename(r['filepath']) for r in results[:3]],
        })

    avg_precision = total_precision / len(queries) if queries else 0
    avg_query_time = total_query_time / len(queries) if queries else 0

    print(f"\n--- Summary ---")
    print(f"  Avg Precision@5: {avg_precision:.1%}")
    print(f"  Avg Query Time:  {avg_query_time:.1f}ms")
    print(f"  Total Chunks:    {total_chunks:,}")
    print(f"  Bucket RAM:      {stats['bucket_wave_ram_kb']:.1f} KB")

    return {
        'name': name,
        'mode': mode,
        'total_chunks': total_chunks,
        'ingest_time_s': round(ingest_time, 2),
        'chunks_per_sec': round(total_chunks / ingest_time) if ingest_time > 0 else 0,
        'num_buckets': stats['num_buckets'],
        'bucket_ram_kb': stats['bucket_wave_ram_kb'],
        'avg_precision_at_5': round(avg_precision, 3),
        'avg_query_time_ms': round(avg_query_time, 3),
        'queries': query_results,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = []

    # 1. Keith codebase (small — full validation)
    r = run_benchmark("Keith Codebase", KEITH_ROOT, KEITH_QUERIES, mode='native')
    if r:
        results.append(r)

    # 2. Linux kernel — subset first (5000 files)
    r = run_benchmark("Linux Kernel (5k files)", LINUX_ROOT, LINUX_QUERIES,
                      mode='native', max_files=5000)
    if r:
        results.append(r)

    # 3. Linux kernel — full run (all 63k files)
    print("\n\nFull Linux kernel ingest starting — this tests the scalability claim...")
    r = run_benchmark("Linux Kernel (FULL)", LINUX_ROOT, LINUX_QUERIES,
                      mode='native', max_files=0)
    if r:
        results.append(r)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print final comparison table
    if results:
        print(f"\n{'='*80}")
        print(f"  FINAL COMPARISON")
        print(f"{'='*80}")
        print(f"  {'Benchmark':<30} {'Chunks':>10} {'Ingest':>10} {'C/sec':>10} {'Query':>10} {'P@5':>8} {'RAM':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
        for r in results:
            print(f"  {r['name']:<30} {r['total_chunks']:>10,} {r['ingest_time_s']:>9.1f}s "
                  f"{r['chunks_per_sec']:>10,} {r['avg_query_time_ms']:>8.1f}ms "
                  f"{r['avg_precision_at_5']:>7.1%} {r['bucket_ram_kb']:>8.1f}KB")
