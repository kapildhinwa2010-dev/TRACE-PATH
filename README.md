# TRACE-PATH

[![Stars](https://img.shields.io/github/stars/kapildhinwa2010-dev/TRACE-PATH?style=flat-square)](https://github.com/kapildhinwa2010-dev/TRACE-PATH/stargazers) [![Forks](https://img.shields.io/github/forks/kapildhinwa2010-dev/TRACE-PATH?style=flat-square)](https://github.com/kapildhinwa2010-dev/TRACE-PATH/network/members) [![Issues](https://img.shields.io/github/issues/kapildhinwa2010-dev/TRACE-PATH?style=flat-square)](https://github.com/kapildhinwa2010-dev/TRACE-PATH/issues) [![Top language](https://img.shields.io/github/languages/top/kapildhinwa2010-dev/TRACE-PATH?style=flat-square)](https://github.com/kapildhinwa2010-dev/TRACE-PATH)

Find which block of a model carries a piece of information, with activation patching.

TRACE-PATH runs a clean input and a corrupted input through any list of stream-additive blocks, then transplants the write of each block from the clean run into the corrupted run, one block at a time. The recovery curve shows exactly where the missing information flows, and the birth index reports the first block whose transplant restores the clean behavior.

## What it does

- Runs clean and corrupted passes through any list of stream-additive blocks.
- Transplants each block write from the clean run into the corrupted run.
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
2. For every block k, replace its write in the corrupted run with the write it produced in the clean run.
3. Normalize each score between the corrupt baseline (0) and the clean baseline (1).

A block whose transplant restores the answer sits on the information path; a block whose transplant changes nothing does not carry the missing information.

## Honest scope

- Blocks must follow stream-additive semantics: state = state + block(state).
- The tracer localizes information flow; it does not prove causation in a trained network.
- Tests use a tiny synthetic block stack, so no model download is needed.

## License

MIT
