import pytest

import infoquality.model as model


@pytest.mark.parametrize("dims", [32, 64, 128, 256])
@pytest.mark.parametrize("max_len", [40, 80, 160])
def test_positional_encoding(dims, max_len):
    pe = model.PositionalEncoding(dims, max_len)
    assert pe.pos_encoding.shape == (1, max_len, dims)
