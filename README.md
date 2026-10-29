# CS540 Tenstorrent Lab

TT-Lang tutorials, programming exercises, and compiler-analysis questions.

Website: https://deep-learning-profiling-tools.github.io/cs540-tenstorrent-lab/

The website is public; this source repository is private. Pushes to `main`
rebuild and publish `dist/` through `.github/workflows/pages.yml`.

## Build and preview

```sh
python3 build_content.py
python3 -m http.server 8765 --directory dist
```

Visit http://localhost:8765/. The build uses the Python standard library and
regenerates the lesson content, offline reading guide, and downloadable kit.

## Contents

- Lab overview and eight numbered chapters covering architecture, TT-Lang,
  matmul, compiler tools, three exercises, and the report.
- Vector addition, ragged sum, and ragged softmax exercises with FP32 inputs
  and outputs, PyTorch references, small examples, and ten test shapes each.
  Row lengths use int64.
- Report requirements worth 100 points, including eight IR questions.
- Upstream MLIR test fixtures and explicitly labeled illustrative compiler traces.

## Source layout

- `course_*.py`: lesson content, figures, exercises, and packaging.
- `build_content.py`: content builder.
- `dist/`: published static website and downloadable materials.
- `teaching_assets/`: teaching kernels, reference implementations, and examples.
- `research/`: retained upstream sources used to develop the material.
- `.openai/hosting.json`: existing Sites deployment identity and static directory.

## Validation

With NumPy installed, run:

```sh
python3 teaching_assets/check_references.py
```

The PyTorch baselines require PyTorch. Device kernels import `ttl` and receive
preallocated device tensors from a compatible runner. They have not been
compiled or validated on Tenstorrent hardware in this project. The optional
full-row ragged kernels support padded widths up to 256; larger cases need
an implementation that processes rows in smaller pieces.

The matmul tutorial and upstream IR fixtures retain their documented data types.
Illustrative compiler traces are labeled as placeholders; they are not captured
compiler output.

## Attribution

Upstream and adapted files retain their license headers and accompanying
license/notice files. See `dist/downloads/provenance.json` for source attribution.
