from __future__ import annotations
import numpy as np
from markov.matrix import Markov_Matrix, State
from markov.tokenizer import Tokenizer

# sample text from a fitted matrix and tokenizer
def generate(matrix: Markov_Matrix, tokenizer: Tokenizer, *, seed: str | None = None, max_tokens: int = 100) -> str:
    if matrix.vocab_size < 1:
        raise ValueError("matrix is not fitted. Call fit() first.")
    if not tokenizer.word_to_index:
        raise ValueError("tokenizer is not fitted. Call fit() first.")

    state = _initial_state(matrix, tokenizer, seed)
    output: list[int] = list(state)

    while len(output) < max_tokens:
        row = matrix.get_row_backoff(state)
        next_idx = int(np.random.choice(matrix.vocab_size, p=row))
        output.append(next_idx)
        state = tuple(state[1:] + (next_idx,))

    return " ".join(tokenizer.decode(output))

def _initial_state(matrix: Markov_Matrix, tokenizer: Tokenizer, seed: str | None) -> State:
    if seed is not None:
        encoded = tokenizer.encode(seed)
        if len(encoded) < matrix.order:
            raise ValueError(
                f"seed must encode to at least {matrix.order} tokens, got {len(encoded)}"
            )
        return tuple(encoded[-matrix.order :])
    observed = matrix.observed_states
    if not observed:
        raise ValueError("no observed states in matrix; cannot start generation")
    index = int(np.random.randint(len(observed)))
    return observed[index]