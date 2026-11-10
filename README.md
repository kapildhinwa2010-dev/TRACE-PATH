# TRACE-PATH

Find the exact block where a piece of information enters the stream.

TRACE-PATH is an activation-patching tracer for stream-additive models. It runs a clean input and a corrupted input, then continues the model from the clean intermediate state at every depth. The recovery curve shows at which depth the missing information is restored, and the birth index reports where the fact first becomes sufficient.

## What it does

- Runs clean and corrupted passes through any list of stream-additive blocks.
- Patches the clean state back in at every depth.
- Returns a normalized recovery curve, clean and corrupt scores, and the birth index.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from trace_path.trace import trace_blocks

blocks = [lambda y: y, lambda y: y * 0.5, lambda y: np.roll(y, 1)]
x_clean = np.array([1.0, 0.0, 0.0, 0.0])
x_corrupt = np.array([0.0, 0.0, 0.0, 1.0])
score = lambda out: float(out[0])

result = trace_blocks(blocks, x_clean, x_corrupt, score)
print(result["recovery_curve"])
print(result["birth_index"])
```

## How it works

1. Score the clean run and the corrupted run.
2. For every depth k, continue the model from the clean state after k blocks.
3. Normalize each score between the corrupt baseline (0) and the clean baseline (1).

The first depth that reaches the threshold is the birth index: the information is present in the clean stream from that point onward.

## Honest scope

- Blocks must follow stream-additive semantics: state = state + block(state).
- The tracer localizes information flow; it does not prove causation in a trained network.
- Tests use a tiny synthetic block stack, so no model download is needed.

## License

MIT
