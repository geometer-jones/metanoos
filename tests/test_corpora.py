from pathlib import Path

import pytest

from metanoos import (
    available_corpora,
    corpus_path,
    get_corpus,
    read_corpus_readme,
    read_corpus_sources,
    read_corpus_text,
)


def test_available_corpora_match_imported_registry() -> None:
    assert [corpus.name for corpus in available_corpora()] == [
        "chinese_classics",
        "greek_classics",
        "lisp_math",
    ]


def test_corpus_path_exposes_expected_layout() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for name in ("chinese_classics", "greek_classics", "lisp_math"):
        corpus = get_corpus(name)

        with corpus_path(corpus.name) as path:
            assert isinstance(path, Path)
            assert path.is_dir()
            assert path == repo_root / "corpora" / corpus.name
            assert (path / corpus.combined_filename).is_file()
            assert (path / corpus.readme_filename).is_file()
            assert (path / corpus.sources_filename).is_file()


def test_corpus_text_and_readme_are_nonempty() -> None:
    for name in ("chinese_classics", "greek_classics", "lisp_math"):
        text = read_corpus_text(name)
        readme = read_corpus_readme(name)

        assert len(text) > 1000
        assert len(readme) > 10


def test_greek_corpus_has_provenance_rows() -> None:
    sources = read_corpus_sources("greek_classics").strip().splitlines()

    assert sources[0] == "author\tebook_id\ttitle\turl\tfilename"
    assert len(sources) > 5


def test_lisp_math_corpus_has_stage_provenance_and_examples() -> None:
    sources = read_corpus_sources("lisp_math").strip().splitlines()

    assert sources[0] == "stage\tkind\texamples\tcharacters"
    assert {line.split("\t")[0] for line in sources[1:]} == {"1", "2", "3", "4", "5", "6", "7"}

    examples = [block for block in read_corpus_text("lisp_math").split("\n\n") if block.strip()]
    assert len(examples) > 1000

    for block in examples[:50]:
        lines = block.splitlines()
        assert len(lines) >= 2
        assert all(line.strip() for line in lines[:2])


def test_unknown_corpus_error_lists_available_names() -> None:
    with pytest.raises(KeyError, match="chinese_classics, greek_classics, lisp_math"):
        get_corpus("missing")
