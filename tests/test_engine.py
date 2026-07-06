"""Test suite for the Markov engine core."""
from pathlib import Path

import numpy as np
import pytest

from markov.generator import generate
from markov.loader import load_text
from markov.matrix import Markov_Matrix
from markov.tokenizer import Tokenizer

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_input.txt"


@pytest.fixture()
def corpus():
    return load_text(str(SAMPLE))


@pytest.fixture()
def fitted(corpus):
    tokenizer = Tokenizer()
    indices = tokenizer.fit_encode(corpus)
    return tokenizer, indices


def _fit_matrix(indices, vocab_size, order):
    matrix = Markov_Matrix(order)
    matrix.fit(indices, vocab_size)
    return matrix


def test_tokenizer_roundtrip(fitted):
    tokenizer, indices = fitted
    decoded = tokenizer.decode(indices)
    assert len(decoded) == len(indices)
    assert all(tokenizer.word_to_index[w] == i for w, i in zip(decoded, indices))


def test_order1_rows_are_distributions(fitted):
    tokenizer, indices = fitted
    matrix = _fit_matrix(indices, tokenizer.vocab_size, 1)
    for state in matrix.observed_states:
        row = matrix.get_row(state)
        assert row.shape == (matrix.vocab_size,)
        assert np.isclose(row.sum(), 1.0)
        assert (row >= 0).all()


def test_order1_generation_only_emits_observed_bigrams(fitted):
    tokenizer, indices = fitted
    matrix = _fit_matrix(indices, tokenizer.vocab_size, 1)
    observed_bigrams = {(indices[i], indices[i + 1]) for i in range(len(indices) - 1)}
    np.random.seed(7)
    text = generate(matrix, tokenizer, max_tokens=80)
    out = tokenizer.encode(text)
    pairs = list(zip(out, out[1:]))
    # the final corpus token may be a dead end; every other step must be real
    unseen = [p for p in pairs if p not in observed_bigrams]
    assert len(unseen) <= 1


def test_higher_order_backoff_never_goes_uniform(fitted):
    """Regression: a dead-end state must back off, not return uniform noise."""
    tokenizer, indices = fitted
    matrix = _fit_matrix(indices, tokenizer.vocab_size, 2)
    # build a state that never occurs in the corpus but whose last token does
    common_token = indices[0]
    bogus_state = (matrix.vocab_size - 1, common_token)
    row = matrix.get_row_backoff(bogus_state)
    uniform = np.full(matrix.vocab_size, 1.0 / matrix.vocab_size)
    assert not np.allclose(row, uniform), "backoff should use the order-1 context"
    assert np.isclose(row.sum(), 1.0)


def test_higher_order_generation_stays_on_corpus_bigrams(fitted):
    tokenizer, indices = fitted
    matrix = _fit_matrix(indices, tokenizer.vocab_size, 3)
    observed_bigrams = {(indices[i], indices[i + 1]) for i in range(len(indices) - 1)}
    np.random.seed(11)
    text = generate(matrix, tokenizer, max_tokens=80)
    out = tokenizer.encode(text)
    pairs = list(zip(out, out[1:]))
    unseen = [p for p in pairs if p not in observed_bigrams]
    assert len(unseen) <= 1


def test_unfitted_matrix_raises():
    matrix = Markov_Matrix(1)
    tokenizer = Tokenizer()
    with pytest.raises(ValueError):
        generate(matrix, tokenizer)


def test_order_must_be_positive():
    with pytest.raises(ValueError):
        Markov_Matrix(0)
