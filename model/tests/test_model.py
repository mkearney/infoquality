import pytest
import torch

from infoquality.hyperparameters import HyperParameters
from infoquality.model import Model, build_config


def test_build_config_labels_and_dropout(hp, label_map):
    config = build_config(hp, label_map)
    assert config.num_labels == 3
    assert config.label2id == label_map
    assert config.id2label == {0: "low", 1: "medium", 2: "high"}
    # DistilBERT's classification head dropout
    assert config.seq_classif_dropout == pytest.approx(0.3)
    # other dropout settings are left untouched
    assert config.dropout != pytest.approx(0.3)


def test_build_config_architecture_overrides(hp, label_map):
    default = build_config(hp, label_map)
    config = build_config(
        hp.model_copy(update={"num_layers": 1, "num_dims": 8}), label_map
    )
    assert config.n_layers == 1
    assert config.dim == 8
    assert config.hidden_dim == 32
    # keys that merely contain "dim" are not overridden
    assert config.max_position_embeddings == default.max_position_embeddings


def test_build_config_no_overrides_by_default(hp, label_map):
    config = build_config(hp, label_map)
    reference = build_config(hp.model_copy(update={"dropout": 0.0}), label_map)
    assert config.n_layers == reference.n_layers
    assert config.dim == reference.dim


def test_default_label_map(hp):
    model = Model(hp.model_copy(update={"num_classes": 2}))
    assert model.label_map == {"0": 0, "1": 1}


@pytest.mark.parametrize(
    "label_map",
    [{"a": 0, "b": 1}, {"a": 0, "b": 1, "c": 3}],
)
def test_invalid_label_map(hp, label_map):
    with pytest.raises(ValueError):
        Model(hp, label_map=label_map)


def test_version_prefix(model, hp):
    assert model.version.startswith(f"{hp.version}.")


@pytest.mark.parametrize("num_tokens", [1, 5, 50])
def test_preprocess_pads_and_truncates(model, num_tokens):
    inputs = model.preprocess([" ".join(["a"] * num_tokens), "b"])
    assert inputs["input_ids"].shape == (2, model.max_len)
    assert inputs["attention_mask"].shape == (2, model.max_len)
    # +2 for the [CLS] and [SEP] tokens
    expected = min(num_tokens + 2, model.max_len)
    assert int(inputs["attention_mask"][0].sum()) == expected


def test_forward_shapes(model):
    with torch.no_grad():
        raw = model.forward_raw(["hello world", "another message"])
        inputs = model.preprocess(["hello world", "another message"])
        logits = model(**inputs, targets=torch.tensor([0, 1]))
    assert raw.shape == (2, 3)
    torch.testing.assert_close(raw, logits)


def test_hyperparameters_reject_unknown_best_metric():
    with pytest.raises(ValueError):
        HyperParameters(best_metric="f1")
