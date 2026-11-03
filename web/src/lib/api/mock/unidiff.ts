// A small unified-diff writer for the mock: it turns before/after file
// contents into the `git diff` text GET /diff returns, so the mock's diffs
// are always well-formed. Line-level LCS; fine for the small files it sees.

type Op = { t: " " | "-" | "+"; line: string };

function lines(s: string): string[] {
  if (s === "") return [];
  const out = s.split("\n");
  if (out[out.length - 1] === "") out.pop();
  return out;
}

function diffOps(a: string[], b: string[]): Op[] {
  const n = a.length;
  const m = b.length;
  // lcs[i][j] = LCS length of a[i:] and b[j:].
  const lcs: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i]![j] = a[i] === b[j] ? lcs[i + 1]![j + 1]! + 1 : Math.max(lcs[i + 1]![j]!, lcs[i]![j + 1]!);
    }
  }
  const ops: Op[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      ops.push({ t: " ", line: a[i]! });
      i++;
      j++;
    } else if (lcs[i + 1]![j]! >= lcs[i]![j + 1]!) {
      ops.push({ t: "-", line: a[i++]! });
    } else {
      ops.push({ t: "+", line: b[j++]! });
    }
  }
  while (i < n) ops.push({ t: "-", line: a[i++]! });
  while (j < m) ops.push({ t: "+", line: b[j++]! });
  return ops;
}

function shortHash(s: string): string {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(16).padStart(8, "0").slice(0, 7);
}

/**
 * The diff of one file. `before` null means the file is new; `after` null
 * means it was deleted. Returns "" when nothing changed.
 */
export function fileDiff(path: string, before: string | null, after: string | null, context = 3): string {
  if (before === after) return "";
  const ops = diffOps(lines(before ?? ""), lines(after ?? ""));
  const changed = ops.map((o, k) => (o.t === " " ? -1 : k)).filter((k) => k >= 0);
  if (changed.length === 0) return "";

  // Group changes into hunks, merging ones whose context would overlap.
  const ranges: [number, number][] = [];
  for (const k of changed) {
    const lo = Math.max(0, k - context);
    const hi = Math.min(ops.length - 1, k + context);
    const last = ranges[ranges.length - 1];
    if (last && lo <= last[1] + 1) last[1] = Math.max(last[1], hi);
    else ranges.push([lo, hi]);
  }

  // Old and new line numbers before each op.
  const oldAt: number[] = [];
  const newAt: number[] = [];
  let o = 1;
  let nn = 1;
  for (const op of ops) {
    oldAt.push(o);
    newAt.push(nn);
    if (op.t !== "+") o++;
    if (op.t !== "-") nn++;
  }

  const out: string[] = [`diff --git a/${path} b/${path}`];
  if (before === null) out.push("new file mode 100644");
  if (after === null) out.push("deleted file mode 100644");
  out.push(`index ${before === null ? "0000000" : shortHash(before)}..${after === null ? "0000000" : shortHash(after)}`);
  out.push(before === null ? "--- /dev/null" : `--- a/${path}`);
  out.push(after === null ? "+++ /dev/null" : `+++ b/${path}`);

  for (const [lo, hi] of ranges) {
    const slice = ops.slice(lo, hi + 1);
    const oldLen = slice.filter((x) => x.t !== "+").length;
    const newLen = slice.filter((x) => x.t !== "-").length;
    const oldStart = oldLen === 0 ? oldAt[lo]! - 1 : oldAt[lo]!;
    const newStart = newLen === 0 ? newAt[lo]! - 1 : newAt[lo]!;
    out.push(`@@ -${oldStart},${oldLen} +${newStart},${newLen} @@`);
    for (const x of slice) out.push(`${x.t}${x.line}`);
  }
  return `${out.join("\n")}\n`;
}
