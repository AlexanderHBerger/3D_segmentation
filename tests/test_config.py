import pytest

from config import Config, apply_overrides


def test_overrides_parse_literals_and_dict_fields():
    c = apply_overrides(Config(), ["training.batch_size=4", "data.patch_size=(64, 64)",
                                   "training.loss_kwargs.weight=0.1", "training.loss_kwargs.penalty_matched=ce"])
    assert c.training.batch_size == 4
    assert c.data.patch_size == (64, 64)
    assert c.training.loss_kwargs == {"weight": 0.1, "penalty_matched": "ce"}


def test_unknown_field_is_an_error():
    with pytest.raises(KeyError):
        apply_overrides(Config(), ["training.no_such_field=1"])


def test_dict_round_trip_ignores_unknown_keys():
    c = Config()
    c.training.loss_kwargs = {"alpha": 0.5}
    d = c.to_dict()
    d["training"]["removed_option"] = 1
    assert Config.from_dict(d) == c
