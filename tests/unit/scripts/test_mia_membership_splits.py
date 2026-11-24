from __future__ import annotations

import numpy as np

from scripts.run_mia import _membership_samples


def test_mia_members_are_train_rows_and_nonmembers_are_held_out_rows() -> None:
    x_train = np.arange(12, dtype=np.float32).reshape(6, 2)
    x_eval = np.arange(100, 112, dtype=np.float32).reshape(6, 2)
    y_train = {"task": np.arange(6)}
    y_eval = {"task": np.arange(10, 16)}

    (x_members, y_members), (x_nonmembers, y_nonmembers) = _membership_samples(
        x_train, y_train, x_eval, y_eval, requested_members=4, requested_nonmembers=3
    )

    np.testing.assert_array_equal(x_members, x_train[:4])
    np.testing.assert_array_equal(y_members["task"], y_train["task"][:4])
    np.testing.assert_array_equal(x_nonmembers, x_eval[:3])
    np.testing.assert_array_equal(y_nonmembers["task"], y_eval["task"][:3])
    assert not np.intersect1d(x_members[:, 0], x_nonmembers[:, 0]).size
