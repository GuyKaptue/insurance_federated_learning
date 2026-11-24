# Privacy mechanism and accounting assumptions

This note describes the synthetic four-client implementation and the C6 smoke evidence. It is not a deployment guarantee or a legal privacy assessment.

## Local update mechanism

When Local DP is enabled, Flower's `LocalDpMod` clips each client's model update to the configured L2 norm, then adds independent Gaussian noise before the update reaches the server.

- Clip norm: `C = 1.0`.
- Noise standard deviation: `1.0` per parameter coordinate.
- Noise multiplier: `sigma = noise_std / C = 1.0`.
- Failure probability: `delta = 1e-5`.
- Accountant event: one Gaussian release with `sigma = 1.0` per fit round.

The mechanism is local because the client randomizes its update before sending it. Central DP would add noise to the aggregated update at the server; this code does not use that mechanism. Secure aggregation is not wired end-to-end through the Flower app path, so no end-to-end SecAgg+ claim is made.

## Adjacency and accountant

`PrivacyConfig.neighboring_relation` is `add_or_remove_one`. For this accounting configuration, the adjacency unit is one client's clipped update: compare no update with one update whose L2 norm is at most `C`. The accountant uses sensitivity `C` and `sigma = noise_std / C`.

This convention does not cover replacement adjacency. Replacing one client's update with another can have sensitivity up to `2C`; the current `sigma = 1` accounting must not be presented as a replacement-adjacency guarantee. The event is client/update-level, not a guarantee for an individual row inside a client's local dataset.

The headline accountant uses `q = 1` (all four clients are selected in the demonstrated run). No subsampling amplification is claimed. The code's amplified epsilon is unavailable and is not used as a guarantee. The four-client population is too small to support a meaningful amplification claim here.

## Measured C6 state

The synthetic C6 smoke run used clip norm `1.0`, noise standard deviation `1.0`, multiplier `1.0`, delta `1e-5`, and an epsilon cap of `8.0`. Two fit releases were accounted before dispatch. At two releases, spent epsilon was `7.0774`; the projected third release was `9.0100`, so the gate refused it. Resume restored the same two-release state. This demonstrated accountant and gate continuity, not model-weight checkpoint restore.

## Limits

- A single-host, four-client simulation is not multi-organization evidence.
- The accountant tracks model-update Gaussian events; it does not make per-client metrics, `num_examples`, audit metadata, or every side channel private.
- Local DP is disabled by default. No privacy guarantee is claimed when it is disabled.
- `reports/dp_utility_curve.json` measures one local release per utility point on synthetic v2 data. It does not claim cumulative epsilon for repeated releases. The observed utility loss is severe; round-1 diagnosis reports the raw CLV target/loss scale and the resulting update clipping in `docs/evidence/round1_loss_diagnosis_2026-11-21.json`.
- This configuration and its accounting convention need review before use with real data or a different adjacency unit, client count, sampling policy, or release mechanism.

## References

- Mironov, I. “Rényi Differential Privacy.” 2017.
- Flower 1.31 `LocalDpMod` implementation and `dp-accounting` RDP event API, as used by `src/insurance_fl/fl/dp.py` and `src/insurance_fl/fl/privacy.py`.
