# Lisp Math Corpus

Deterministic synthetic pretraining corpus for the repository's custom Lisp
mathematics dialect.

- Generator: `scripts/generate_lisp_math_corpus.py`
- Seed: `20260423`
- Examples: `268,205`
- Characters: `10,000,039`
- Format: prompt line, answer/proof line(s), blank line
- Dialect source of truth: `src/reciprocator/rl/lisp_eval.py`

The corpus follows the RL mathematical curriculum:

1. Arithmetic: `+`, `-`, `*`, `/`, `mod`, `expt`
2. Local bindings, comparisons, booleans: `let`, `=`, `<`, `>`, `<=`, `>=`, `and`, `or`, `not`
3. Code as data: `quote`, `list`, `eval`
4. Conditionals and functions: `if`, `cond`, `define`, `lambda`
5. Symbolic algebra: `simplify`, `expand`, `substitute`, `differentiate`
6. Proof construction: validated `(step rule input output)` sequences
7. Collections and higher-order operations: vectors, maps, `get`, `count`, `conj`, `assoc`, `mapv`, `filterv`, `reduce`, `->`, `->>`, `comp`, `partial`

Example:

```lisp
(+ 1 (* 2 3))
7

(prove x from (+ x 0))
(step simplify-add-zero (+ x 0) x)
```
