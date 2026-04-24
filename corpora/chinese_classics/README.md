## Chinese Classics Corpus

Classical Chinese philosophy and literature training corpus sourced from the
`da-xue` project, originally scraped from ctext.org.

- Source family: ctext.org (Chinese Text Project)
- Cleaning: structured JSON extraction, reading unit text only
- Bundled assets:
  - `chinese_classics_combined.txt`: combined training text (reading units)
  - `sources.tsv`: book, chapter id, title, and character count

Works included:

- 論語集注 (Analects)
- 孟子 (Mencius)
- 大學 (Great Learning)
- 中庸 (Doctrine of the Mean)
- 道德經 (Daodejing)
- 孫子兵法 (Art of War)
- 三國演義 (Romance of the Three Kingdoms)
- 成語Catalog (Chengyu catalog)
- 千字文 (Thousand Character Classic)
- 三字經 (Three Character Classic)

Access from Python:

```python
from reciprocator import read_corpus_text

text = read_corpus_text("chinese_classics")
```
