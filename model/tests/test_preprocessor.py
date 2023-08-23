import infoquality.preprocessor as pp
import pytest


@pytest.fixture
def preprocessor():
    return pp.Preprocessor()


@pytest.fixture
def message():
    def func(num_tokens: int) -> str:
        return " ".join(["a"] * num_tokens)

    return func


@pytest.mark.parametrize("num_tokens", [i**3 for i in range(1, 10)])
def test_preprocessor(preprocessor, message, num_tokens):
    assert len(message(num_tokens).split()) == num_tokens
    assert (len(preprocessor([message(num_tokens)])[0]) - 2) == min(
        num_tokens, preprocessor.max_len - 2
    )
