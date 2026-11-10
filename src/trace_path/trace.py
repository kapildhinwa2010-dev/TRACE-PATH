"""Activation-patching tracer: localize which block carries a fact.

Given a clean input and a corrupted input, TRACE-PATH swaps the
clean intermediate state back in at every depth and measures how
much of the clean behavior recovers. The recovery curve shows at
which depth the information enters the stream.

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


def trace_blocks(blocks, x_clean, x_corrupt, score_fn, threshold=0.5):
    """Patch the clean stream in at every depth and measure recovery.

    ``score_fn`` maps a final state to a scalar. The recovery value
    is normalized so that the corrupt run is 0 and the clean run is 1.
    ``birth_index`` is the smallest depth at which recovery reaches
    ``threshold``. curve[0] is the corrupt baseline and curve[len(blocks)]
    is the clean baseline.
    """
    if not blocks:
        raise ValueError("blocks must be a non-empty list")
    n = len(blocks)
    clean_out = run_stream(blocks, x_clean)
    corrupt_out = run_stream(blocks, x_corrupt)
    s_clean = float(score_fn(clean_out))
    s_corrupt = float(score_fn(corrupt_out))
    denom = s_clean - s_corrupt

    states = [np.asarray(x_clean, dtype=float)]
    x = states[0]
    for b in blocks:
        x = x + b(x)
        states.append(x)

    curve = []
    for k in range(n + 1):
        if abs(denom) < 1e-12:
            curve.append(0.0)
        else:
            out_k = run_stream(blocks, states[k], start=k)
            curve.append((float(score_fn(out_k)) - s_corrupt) / denom)

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