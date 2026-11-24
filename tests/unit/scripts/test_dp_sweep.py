from __future__ import annotations

import math

import pytest

from scripts.dp_sweep import _epsilon_for_noise_multiplier


@pytest.mark.parametrize("noise_multiplier", [0.5, 1.0, 2.0])
def test_dp_sweep_epsilon_maps_to_requested_noise_multiplier(noise_multiplier: float) -> None:
    delta = 1e-5
    clipping_norm = 1.0
    sensitivity = clipping_norm
    epsilon = _epsilon_for_noise_multiplier(noise_multiplier, sensitivity, clipping_norm, delta)
    actual_multiplier = (
        sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / (clipping_norm * epsilon)
    )

    assert actual_multiplier == pytest.approx(noise_multiplier)
