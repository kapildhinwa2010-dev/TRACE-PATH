import numpy as np
import pytest

from trace_path.trace import run_stream, trace_blocks

D = 4
FACT = np.zeros(D)
FACT[3] = 1.0


def make_blocks():
    # block 0: identity write (copies the marker into the stream)
    b0 = lambda y: y
    # block 1: writes the fact unconditionally
    b1 = lambda y: FACT.copy()
    # block 2: routes the fact to slot 1, but only if the marker is present
    def b2(y):
        out = np.zeros(D)
        if y[0] > 0.5:
            out[1] = 3.0 * y[3]
        return out

    return [b0, b1, b2]


def score_fn(out):
    return float(out[1])


def x_clean():
    v = np.zeros(D)
    v[0] = 1.0
    return v


def x_corrupt():
    v = np.zeros(D)
    v[2] = 1.0
    return v


def test_run_stream():
    blocks = make_blocks()
    out = run_stream(blocks, x_clean())
    # stream: e0 + e0 + fact + routed = [2, 3, 0, 1]
    assert np.allclose(out, [2.0, 3.0, 0.0, 1.0])


def test_trace_curve():
    blocks = make_blocks()
    r = trace_blocks(blocks, x_clean(), x_corrupt(), score_fn)
    assert r["clean_score"] == pytest.approx(3.0)
    assert r["corrupt_score"] == pytest.approx(0.0)
    assert r["recovery_curve"][0] == pytest.approx(1.0)
    assert r["recovery_curve"][1] == pytest.approx(0.0)
    assert r["recovery_curve"][2] == pytest.approx(1.0)
    assert r["birth_index"] == 0


def test_trace_uninformative_inputs():
    blocks = make_blocks()
    r = trace_blocks(blocks, x_clean(), x_clean().copy(), score_fn)
    assert all(v == 0.0 for v in r["recovery_curve"])
    assert r["birth_index"] is None


def test_threshold_changes_birth():
    blocks = make_blocks()
    r = trace_blocks(blocks, x_clean(), x_corrupt(), score_fn, threshold=1.5)
    assert r["birth_index"] is None


def test_empty_blocks_error():
    with pytest.raises(ValueError):
        trace_blocks([], np.zeros(D), np.zeros(D), score_fn)