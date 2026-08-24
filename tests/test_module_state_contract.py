import numpy as np
import pytest

from micro_dl import Buffer, Cache, Module, Parameter


class Leaf(Module):
    def __init__(self, shared=None):
        super().__init__()
        self.weight = shared or Parameter([[1.0, 2.0]])
        self.register_buffer("running", Buffer([3.0]))
        self.cache = Cache("temporary")


class Tree(Module):
    def __init__(self):
        super().__init__()
        shared = Parameter([[1.0, 2.0]])
        self.left = Leaf(shared)
        self.right = Leaf(shared)


def test_shared_parameter_has_first_canonical_name_and_buffer_is_not_parameter():
    model = Tree()

    assert [name for name, _ in model.named_parameters()] == ["left.weight"]
    assert [name for name, _ in model.named_buffers()] == ["left.running", "right.running"]
    assert all("cache" not in name for name in model.state_dict())


def test_mode_propagates_through_explicit_module_registry():
    model = Tree().eval()
    assert model.training is False
    assert model.left.training is False
    assert model.right.training is False
    model.train()
    assert all(module.training for module in (model, model.left, model.right))


def test_state_round_trip_is_independent_and_strict():
    model = Tree()
    state = model.state_dict()
    state["left.weight"][0, 0] = 7.0
    np.testing.assert_allclose(model.left.weight.to_numpy(), [[1.0, 2.0]])
    model.load_state_dict(state)
    np.testing.assert_allclose(model.left.weight.to_numpy(), [[7.0, 2.0]])
    assert model.right.weight is model.left.weight

    bad = dict(state)
    bad.pop("left.running")
    bad["extra"] = np.array([1.0])
    bad["left.weight"] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="missing=.*left.running.*unexpected=.*extra.*shape=.*left.weight"):
        model.load_state_dict(bad)


def test_non_strict_load_reports_all_incompatibilities():
    model = Tree()
    result = model.load_state_dict({"extra": np.array([1.0])}, strict=False)
    assert "left.weight" in result.missing_keys
    assert result.unexpected_keys == ("extra",)


def test_parameter_and_buffer_roles_cannot_be_reversed():
    with pytest.raises(ValueError, match="Parameter"):
        Parameter([1.0]).requires_grad = False
    with pytest.raises(ValueError, match="Buffer"):
        Buffer([1.0]).requires_grad = True
