# House Style — Signals to Transformers

This is the editor-facing style guide. The reader never sees this file.
Use it when writing new chapters or polishing existing ones.

The goal is **consistent editorial restraint**: every page should feel
like it came out of the same book. Use the patterns below; resist the
urge to invent new ones.

---

## 1. Palette

Borrowed from the cover. Reuse it everywhere — Mermaid diagrams, matplotlib
plots, admonition tints (when overriding), and any inline SVG.

| Role | Hex | Use for |
|---|---|---|
| Background | `#F5F1E8` | Figure backgrounds, paper feel |
| Body | `#1A1A1A` | Text, primary line work, axis labels |
| Primary accent | `#2A3F8F` | Main signal / model curves, indigo borders |
| Secondary accent | `#C45A3A` | Noise, residuals, "look here" highlights |
| Support 1 | `#5B7553` | Third series, sage green |
| Support 2 | `#8B6F47` | Fourth series, warm tan |

Do not introduce new accent colours without updating this table.

---

## 2. Figures (matplotlib)

Every Python plotting block in the book should start with the project
stylesheet.

```python
import matplotlib.pyplot as plt
plt.style.use('book/figures/style.mplstyle')
```

This gives you the warm off-white background, serif text, indigo + rust
colour cycle, and clean spines (top/right hidden) automatically.

**Figure rules of thumb:**

- One idea per figure. If you have two ideas, make two figures.
- Aspect ratio: `figsize=(7, 4.2)` is the default (set in the stylesheet).
  Use `(7, 7)` for square plots like 2D Nyquist diagrams.
- Always label axes with units. `time t (s)` not `t`.
- Titles state *what is shown*, not what to think. "MSE vs sampling
  rate" not "Why oversampling helps."
- Every figure gets a markdown caption immediately below it. Captions
  are for the reader who only looks at the pictures — make them
  self-contained.

```markdown
![Caption that stands alone — describes what the figure shows and why
it matters in one or two sentences.](../../figures/ch01_nyquist_mse.png){#fig:ch01-nyquist-mse}
```

The `{#fig:...}` label lets you cross-reference: `[](#fig:ch01-nyquist-mse)`
renders as "Figure 1.4" or similar (auto-numbered, see §6).

---

## 3. Admonitions

Stop using blockquotes for callouts. Use typed admonitions — they render
as proper coloured strip-boxes and are semantic, not decorative.

| Pattern | When to use |
|---|---|
| `tip` | Useful intuition or a shortcut the reader can carry forward |
| `note` | Aside that's helpful but not essential to the main thread |
| `important` | Worked example, MVTec callout, or a fact the chapter hinges on |
| `warning` | Common pitfall, easy mistake, fragile assumption |
| `caution` | Subtle gotcha that doesn't fail loudly |
| `seealso` | Forward/backward reference to another chapter or section |

### Example

```markdown
:::{important} MVTec running example
Average multiple exposures of the same tile patch to suppress sensor noise,
then apply a smoothing filter to separate the slow-varying background
texture from sharp defect edges.
:::

:::{warning} Don't sample exactly at $f_s = 2 f_m$
The Nyquist criterion is *strictly* greater than. Equality is fragile to
phase offset — a critically-sampled cosine and a critically-sampled sine
look identical at the sample times.
:::

:::{tip} Why squared error?
It's the maximum-likelihood estimator under Gaussian noise. The full
derivation lives in Chapter 7.
:::
```

**Box quote pattern** — for named quotes (Box, Tukey, Shannon):

```markdown
:::{epigraph}
All models are wrong; some are useful.

— George Box
:::
```

### Migrating from blockquotes

In existing chapters, convert by intent — not blindly:

- "Intuition: …" → `:::{tip}`
- "MVTec example: …" → `:::{important}`
- "Note that…" or "(Don't forget…)" → `:::{note}` or `:::{caution}`
- "Common pitfall:" → `:::{warning}`
- Definition / theorem statements → `:::{prf:definition}` / `:::{prf:theorem}`

Plain `> blockquote` should be reserved for actual quoted text.

---

## 4. Math

- Inline math with `$…$`, display with `$$…$$`. No `\begin{equation}`
  unless you need numbering with a custom label.
- Use the project macros (defined in `myst.yml`):
  - `\E[X]` for expectation
  - `\Var[X]`, `\Cov[X,Y]`
  - `\R`, `\N`, `\Z` for number sets
  - `\norm{x}` for $\|x\|$
  - `\inner{x, y}` for inner product
  - `\given` for conditioning bar
  - `\iid`, `\indep` for sampling and independence
  - `\T` for transpose, `\argmax`, `\argmin`
- Use `\,` for thin spaces in units: `9.81\,\text{m/s}^2`.
- Prefer `\mathrm{}` over `\text{}` for multi-letter operators
  (`\mathrm{MSE}`, `\mathrm{SNR}`).

### Numbered equations

Use `{eq}` references rather than hard-coded numbers:

```markdown
The measurement model is

$$
y = f(x) + \epsilon
$$ (eq:measurement)

…and equation [](#eq:measurement) is what the rest of the book hangs on.
```

### Definitions, theorems, proofs

For formal math content use the `prf` directive family:

```markdown
:::{prf:definition} Sampling at rate $f_s$
Given a continuous signal $x(t)$, the *sampled signal* is the sequence
$x[n] = x(n/f_s)$ for integer $n$.
:::

:::{prf:theorem} Nyquist–Shannon sampling theorem
A bandlimited signal with maximum frequency $f_m$ can be perfectly
reconstructed from samples taken at rate $f_s > 2 f_m$.
:::
```

---

## 5. Diagrams — prefer Mermaid over ASCII

ASCII boxes (`┌─┐` etc.) are fine in plain READMEs read in a terminal,
but on the rendered site they look like a relic. For anything in the
book proper, use Mermaid:

````markdown
```{mermaid}
flowchart TD
    A["Continuous signal $x(t)$"] --> B["Sampler at rate $f_s$"]
    B --> C["Discrete samples $x[n]$"]
    C --> D["Sinc reconstruction"]
    D --> E["Reconstructed $\\hat{x}(t)$"]

    classDef block fill:#F5F1E8,stroke:#2A3F8F,stroke-width:2px,color:#1A1A1A
    class A,B,C,D,E block
```
````

Mermaid styling conventions for this book:

- Box fill: `#F5F1E8` (paper background)
- Box stroke: `#2A3F8F` (indigo) for primary flow, `#C45A3A` (rust) for
  side-tracks or warnings.
- Stroke width: `2px`
- Text colour: `#1A1A1A`
- Use **solid arrows** for sequence/dependency, **dashed arrows** for
  "in parallel" or "optional."

---

## 6. Numbering and cross-references

Auto-numbering is on for figures, equations, and `##` headings (see
`myst.yml`). Use it.

| Target | Label syntax | Reference syntax |
|---|---|---|
| Figure | `{#fig:ch01-nyquist}` after image | `[](#fig:ch01-nyquist)` |
| Equation | `$$ …$$ (eq:measurement)` | `[](#eq:measurement)` |
| Section | `## Title` (auto) | `[](#chapter-1-digitisation)` |
| Theorem | `:::{prf:theorem} …:::` with `:label:` | `[](#thm-nyquist)` |

Never hard-code "Figure 3" or "equation 1.4" — let MyST do it. Numbers
shift when chapters move; references should follow automatically.

---

## 7. Code blocks

- Always specify the language: `` ```python ``, `` ```bash ``, `` ```yaml ``.
- Use four-space indentation in Python (the matplotlib examples in this
  book do this — keep it consistent).
- Comment the *why*, not the *what*. Match Chapter 0's voice.
- Avoid `print(…)` for final outputs; let the figure or expression
  carry the meaning. Use prints only when the chapter genuinely needs
  intermediate values.
- Imports go at the top of each block, even if redundant — every code
  block should be runnable in isolation.

---

## 8. Voice and tone

- Second person, present tense: *"You point a sensor at it…"*
- Short sentences. Em-dashes are allowed and encouraged.
- Define a term in **bold** the first time it appears. After that, use
  it freely.
- "We" for shared reasoning ("we now have…"), "you" for action
  ("you measure…"). Never "I."
- No exclamation marks. No emojis (book-wide rule).
- Footnotes are fine for dense math — but if a footnote runs more than
  two lines, promote it to an admonition.

---

## 9. Chapter scaffolding

Every chapter starts with:

```markdown
# Chapter N — Title
### Optional subtitle

> One-sentence statement of what this chapter buys you.

**Prerequisites:** [Chapter X](…), [Section Y](…)
**You'll be able to:** ⟨one or two specific things⟩

---

## N.1 First section
…
```

And ends with:

```markdown
## Summary

| Concept | Key idea |
|---------|----------|
| … | … |

## Exercises
1. …
2. …

---

**Next →** [Chapter N+1 — …](…)
```

---

## 10. Things to avoid

- Inventing new colours, fonts, or admonition types.
- Putting math inside figure titles. Use the caption.
- Plain blockquotes for callouts (use admonitions).
- ASCII art diagrams in chapter content (use Mermaid).
- Hard-coded "Figure 3" / "Section 2.1" references (use cross-refs).
- Emojis.
- Multiple ideas per figure.
- More than ~3 nested headings (`####` is the limit).

---

If something falls outside this guide, *first* check whether the
existing patterns can absorb it. Only extend the guide when a pattern
will appear in three or more chapters.
