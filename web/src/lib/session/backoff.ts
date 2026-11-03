// Reconnect delays for the event stream (docs/api.md, Client behaviour):
// min(max, base * factor^attempt), scaled by a random factor in [0.5, 1).

export interface BackoffOptions {
  baseMs: number;
  maxMs: number;
  factor: number;
}

export const DEFAULT_BACKOFF: BackoffOptions = { baseMs: 500, maxMs: 15_000, factor: 2 };

/**
 * The delay before reconnect attempt `attempt` (0 for the first retry).
 * `random` returns a number in [0, 1), like Math.random.
 */
export function backoffDelay(
  attempt: number,
  random: () => number = Math.random,
  opts: BackoffOptions = DEFAULT_BACKOFF,
): number {
  const n = Math.max(0, Math.floor(attempt));
  // Cap the exponent so huge attempt counts cannot overflow to Infinity.
  const ceiling = Math.min(opts.maxMs, opts.baseMs * opts.factor ** Math.min(n, 30));
  const r = Math.min(Math.max(random(), 0), 0.999_999);
  return Math.round(ceiling * (0.5 + r * 0.5));
}
