# Signals to Transformers

**Math and Machine Learning Foundations from Measurement to Multimodal AI.**

A single long-form book written for practitioners who want to understand
modern AI from the ground up. The narrative runs from **measurement physics
and noise** (what a sensor actually produces), through **classical signal
processing** (how pixels and features work), into **the mathematical
foundations** (probability, linear algebra, optimization), and up to
**modern deep learning** (CNNs, attention, transformers, vision-language
models, and multimodal AI).

The thread tying it all together is **signals** — physical, digital, and
semantic. Pixels, tokens, audio frames, sensor streams: all signals. The
book treats them that way from chapter 1.

## Who it's for

- Computer vision engineers who want a firm grasp on the math and modern
  architectures.
- ML engineers who want the physical / measurement grounding most ML
  textbooks skip.
- Researchers and graduate students wanting one coherent narrative rather
  than stitching together a probability book, a DSP book, a deep learning
  book, and a transformers tutorial.

## Structure

### Part I — Math Foundations
Probability, linear algebra, and optimization built from scratch with CV
examples on every page. Start here if you need the math before the signal
processing.

### Part II — Signals and Measurement
How a continuous scene becomes a digital array. Sampling, Nyquist,
sensors, pixels, contrast, colour.

### Part III — Why Raw Signals Fail
What goes wrong when you try to do anything useful with raw pixels —
motivates the entire rest of the book.

### Part IV — The Math You Need (applied)
Probability and linear algebra re-contextualised for the CV / ML setting.
Bridges the foundations (Part I) to the architectures (Parts V–VI).

### Part V — Learning from Signals
Convolutions, backprop, CNNs, training dynamics.

### Part VI — Attention and Beyond
Self-attention, transformers, vision transformers, vision-language models,
multimodal AI — where the field is going.

## How this book is built

Each chapter is a paired `.md` (theory, math, pedagogy) + `.py` (runnable
simulation) file. The book renders via MyST Markdown; the scripts run as
standalone Python so readers can reproduce every figure on their own
machine.

### Why `.md + .py` instead of notebooks?

- **Publication-ready** — MyST renders `.md` natively. Cleaner than raw
  `.ipynb`.
- **Reproducible** — `.py` files run fresh every time; no stale cached
  outputs.
- **Reviewable** — git diffs read like prose. Notebook JSON diffs don't.
- **Testable** — standalone scripts drop straight into CI.

Iteration-heavy drafting happens in `.ipynb` elsewhere (e.g.
`~/projects/nn-basics/fundamentals/`). Stable material gets converted to
`.md + .py` and promoted here for publication.

## Project layout

```
cv-ml/
├── README.md                              ← this file
├── myst.yml                               ← book config + TOC
├── pyproject.toml                         ← uv dependencies
├── math/
│   ├── probability/                       ← Part I — probability (complete)
│   ├── linear_algebra/                    ← Part I — planned
│   └── optimization/                      ← Part I — planned
└── book/
    ├── figures/                           ← all static figures
    ├── part2_signals_and_measurement/     ← Part II
    ├── part3_why_raw_signals_fail/        ← Part III
    ├── part4_the_math/                    ← Part IV
    ├── part5_learning_from_signals/       ← Part V
    └── part6_attention_and_beyond/        ← Part VI
```

## Running the code

```bash
cd ~/projects/cv-ml
uv venv
source .venv/bin/activate
uv pip install -e .

# run any part
python math/probability/part4_normal.py
python math/probability/part5_clt.py
```

## Building the book

```bash
# install MyST CLI if you haven't
npm install -g mystmd

# from the project root
myst build --html
# open _build/html/index.html
```

## Status

| Part | Status |
|------|--------|
| Part I — Probability | ✅ complete (part0–part6 + exercises) |
| Part I — Linear Algebra | ⏳ planned (moving from `cnn/tutorials/01b_*`) |
| Part I — Optimization | ⏳ planned |
| Part II — Signals and Measurement | 🟢 chapters drafted (ch01–ch05) |
| Part III — Why Raw Signals Fail | 🟢 ch06 drafted |
| Part IV — The Math You Need | 🟢 ch07–ch08 drafted |
| Part V — Learning from Signals | 🟡 chapter scaffolding (ch09–ch12) |
| Part VI — Attention and Beyond | 🟡 chapter scaffolding (ch13–ch15) |

## License

- Code: MIT
- Content (prose, figures, pedagogy): CC-BY-4.0
