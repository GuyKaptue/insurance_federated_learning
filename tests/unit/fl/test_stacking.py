import numpy as np
import pytest

from insurance_fl.fl import stacking


def test_stacking_exports() -> None:
    assert "fit_stacking_meta" in stacking.__all__


def test_nnls_nonnegative() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    b = rng.normal(size=50)
    y = 2 * a + 1 * b
    w = stacking.fit_stacking_meta([a, b], y)
    assert np.all(w >= 0)
    pred = stacking.predict_stacked([a, b], w)
    assert pred.shape == a.shape


def test_save_meta(tmp_path) -> None:
    w = np.array([1.0, 2.0, 3.0])
    p = stacking.save_stacking_meta(tmp_path / "m.json", w, 0.5)
    assert p.exists()


def test_validate_alpha_valid() -> None:
    import numpy as np

    from insurance_fl.fl.stacking import validate_alpha

    alpha = validate_alpha([0.644, 0.0, 0.321])
    assert np.allclose(alpha, [0.644, 0.0, 0.321])
    assert np.all(alpha >= 0)


def test_validate_alpha_allows_zero_coefficients() -> None:

    from insurance_fl.fl.stacking import validate_alpha

    alpha = validate_alpha([1.0, 0.0, 0.0])
    assert alpha[1] == 0.0
    assert alpha[2] == 0.0


def test_validate_alpha_rejects_all_zero() -> None:

    from insurance_fl.fl.stacking import validate_alpha

    with pytest.raises(ValueError, match="degenerate"):
        validate_alpha([0.0, 0.0, 0.0])


def test_validate_alpha_rejects_negative() -> None:
    from insurance_fl.fl.stacking import validate_alpha

    with pytest.raises(ValueError, match="nonnegative"):
        validate_alpha([0.5, -0.1, 0.3])


def test_validate_alpha_rejects_nan() -> None:

    from insurance_fl.fl.stacking import validate_alpha

    with pytest.raises(ValueError, match="NaN"):
        validate_alpha([0.5, float("nan"), 0.3])


def test_validate_alpha_rejects_empty() -> None:
    from insurance_fl.fl.stacking import validate_alpha

    with pytest.raises(ValueError):
        validate_alpha([])
