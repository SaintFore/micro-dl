"""Tensor public behavior across NumPy and CuPy Backends."""

import numpy as np
import pytest

from micro_dl import Tensor, get_default_backend, use_backend


def _cupy_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, Exception):
        return False


BACKENDS = ["numpy"] + (["cupy"] if _cupy_available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_tensor_uses_selected_backend_and_copies_input(backend):
    source = np.array([1.0, 2.0], dtype=np.float32)
    tensor = Tensor(source, backend=backend)

    source[0] = 99.0

    assert tensor.backend == backend
    np.testing.assert_allclose(tensor.to_numpy(), [1.0, 2.0])


def test_numpy_is_the_default_backend():
    assert get_default_backend() == "numpy"
    assert Tensor([1.0]).backend == "numpy"


def test_backend_context_is_scoped_and_restored():
    before = get_default_backend()

    with use_backend("numpy"):
        assert get_default_backend() == "numpy"
        assert Tensor([1.0]).backend == "numpy"

    assert get_default_backend() == before


def test_python_values_use_float32_but_arrays_keep_dtype():
    assert Tensor([1.0]).dtype == np.dtype(np.float32)
    assert Tensor(np.array([1.0], dtype=np.float64)).dtype == np.dtype(np.float64)
    assert Tensor([1.0], dtype=np.float64).dtype == np.dtype(np.float64)


def test_tensor_copy_preserves_source_backend_by_default():
    source = Tensor([1.0, 2.0], backend="numpy")
    copied = Tensor(source)

    assert copied.backend == source.backend
    assert copied._storage is not source._storage


@pytest.mark.parametrize("backend", BACKENDS)
def test_metadata_item_and_observation_are_backend_neutral(backend):
    tensor = Tensor(np.array([[3.5]], dtype=np.float32), backend=backend)

    assert tensor.shape == (1, 1)
    assert tensor.ndim == 2
    assert tensor.size == 1
    assert tensor.dtype == np.dtype(np.float32)
    assert tensor.item() == pytest.approx(3.5)

    observed = tensor.to_numpy()
    observed[0, 0] = 99.0
    assert tensor.item() == pytest.approx(3.5)


def test_item_rejects_non_scalar_tensor():
    with pytest.raises(ValueError, match="单元素|标量"):
        Tensor([1.0, 2.0]).item()


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown Backend"):
        Tensor([1.0], backend="missing")


def test_cupy_selection_has_clear_unavailable_error():
    if _cupy_available():
        pytest.skip("CuPy and CUDA are available")

    with pytest.raises(RuntimeError, match="CuPy Backend is unavailable"):
        Tensor([1.0], backend="cupy")


@pytest.mark.parametrize("backend", BACKENDS)
def test_operator_storage_can_be_adopted_without_copy(backend):
    source = Tensor([1.0, 2.0], backend=backend)
    storage = source._storage.copy()

    result = Tensor._from_storage(storage, backend=backend)
    storage[0] = 7.0

    assert result._storage is storage
    np.testing.assert_allclose(result.to_numpy(), [7.0, 2.0])
