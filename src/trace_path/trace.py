"""Activation-patching tracer: localize which block carries a fact.

Given a clean input and a corrupted input, TRACE-PATH transplants the
write of one block from the clean run into the corrupted run, for
every block, and measures how much clean behavior recovers. If
transplanting block k recovers the answer, the information flows
through block k.

Blocks are stream-additive callables: state = state + block(state).
Pure NumPy.
"""

import numpy as np


def run_stream(blocks, x, start=0, stop=None):
    """Run state through blocks with stream-additive semantics."""
    if stop is None:
        stop = len(blocks)
    for b in blocks[start:stop]:
        x = x + b(x)
    return x


def _states(blocks, x):
    """Return the stream entering every block, plus the final state."""
    states = [np.asarray(x, dtype=float)]
    current = states[0]
    for b in blocks:
        current = current + b(current)
        states.append(current)
    return states


def trace_blocks(blocks, x_clean, x_corrupt, score_fn, threshold=0.5):
    """Transplant every block write from the clean run into the corrupt run.

    ``score_fn`` maps a final state to a scalar. Recovery values are
    normalized so that the corrupt baseline is 0 and the clean baseline
    is 1. ``recovery_curve[k]`` is the recovery when the write of block
    k is replaced by its clean-run counterpart. ``birth_index`` is the
    first block whose transplant reaches ``threshold``.
    """
    if not blocks:
        raise ValueError("blocks must be a non-empty list")
    n = len(blocks)
    x_clean = np.asarray(x_clean, dtype=float)
    x_corrupt = np.asarray(x_corrupt, dtype=float)

    clean_out = run_stream(blocks, x_clean)
    corrupt_out = run_stream(blocks, x_corrupt)
    s_clean = float(score_fn(clean_out))
    s_corrupt = float(score_fn(corrupt_out))
    denom = s_clean - s_corrupt

    clean_states = _states(blocks, x_clean)

    curve = []
    for k in range(n):
        clean_write = blocks[k](clean_states[k])
        state = x_corrupt
        for i, b in enumerate(blocks):
            write = clean_write if i == k else b(state)
            state = state + write
        s1 = float(score_fn(state))
        if abs(denom) < 1e-12:
            curve.append(0.0)
        else:
            curve.append((s1 - s_corrupt) / denom)

    birth = None
    for k, value in enumerate(curve):
        if value >= threshold:
            birth = k
            break

    return {
        "clean_score": s_clean,
        "corrupt_score": s_corrupt,
        "recovery_curve": curve,
        "birth_index": birth,
    }