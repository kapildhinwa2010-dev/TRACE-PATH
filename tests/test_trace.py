import numpy as np
import pytest

from trace_path.trace import run_stream, trace_blocks

D = 4


def make_blocks():
    # block 0: identity write
    b0 = lambda y: y
    # block 1: writes a fact vector if the marker is present
    marker = np.zeros(D)
    marker[0] = 1.0
    fact = np.zeros(D)
    fact[3] = 1.0

    def b1(y):
        if float(np.dot(y, marker)) > 0.5:
            return fact.copy()
        return np.zeros(D)

    # block 2: routes the fact into output slot 1
    def b2(y):
        out = np.zeros(D)
        out[1] = 3.0 * y[3]
        return out

    return [b0, b1, b2], marker


def score_fn(out):
    return float(out[1])


def test_run_stream():
    blocks, _ = make_blocks()
    x = np.zeros(D)
    x[0] = 1.0
    out = run_stream(blocks, x)
    # stream: x + x + fact + routed = [2, 3, 0, 2]
    assert np.allclose(out, [2.0, 3.0, 0.0, 2.0])


def test_trace_birth():
    blocks, marker = make_blocks()
    x_clean = np.zeros(D)
    x_clean[0] = 1.0
    x_corrupt = np.zeros(D)
    x_corrupt[2] = 1.0
    r = trace_blocks(blocks, x_clean, x_corrupt, score_fn)
    assert r["clean_score"] == pytest.approx(3.0)
    assert r["corrupt_score"] == pytest.approx(0.0)
    assert r["recovery_curve"][0] == pytest.approx(0.0)
    assert r["recovery_curve"][1] == pytest.approx(1.0)
    assert r["recovery_curve"][3] == pytest.approx(1.0)
    assert r["birth_index"] == 1


def test_trace_uninformative_inputs():
    blocks, _ = make_blocks()
    x1 = np.zeros(D)
    x1[0] = 1.0
    r = trace_blocks(blocks, x1, x1.copy(), score_fn)
    # clean and corrupt identical: no signal, flat curve
    assert all(v == 0.0 for v in r["recovery_curve"])
    assert r["birth_index"] is None


def test_threshold_changes_birth():
    blocks, marker = make_blocks()
    x_clean = np.zeros(D)
    x_clean[0] = 1.0
    x_corrupt = np.zeros(D)
    x_corrupt[2] = 1.0
    r = trace_blocks(blocks, x_clean, x_corrupt, score_fn, threshold=0.5)
    assert r["birth_index"] == 1
    # an impossible threshold never fires
    r2 = trace_blocks(blocks, x_clean, x_corrupt, score_fn, threshold=2.0)
    assert r2["birth_index"] is None


def test_empty_blocks_error():
    with pytest.raises(ValueError):
        trace_blocks([], np.zeros(D), np.zeros(D), score_fn)