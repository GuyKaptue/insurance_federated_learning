"""Flower ClientApp (Layer 1 / Task 0): local training on one insurance client.

One simulated supernode maps to one of the 4 real client identities
(``node_id % 4`` — stable within a ``flwr run`` simulation session;
Task 11 revisits partitioning for the integration test).

Each client is READ-ONLY against the pre-generated smoke dataset
(``python -m insurance_fl.fl.task --prepare`` runs first).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context, Scalar

from insurance_fl.fl import task as fl_task
from insurance_fl.models.multi_task_nn import MultiTaskNN

log = logging.getLogger(__name__)


class InsuranceNNClient(NumPyClient):
    """Trains the MultiTaskNN on one client's local partition."""

    def __init__(
        self,
        net: MultiTaskNN,
        x: np.ndarray,
        y: dict[str, np.ndarray],
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        super().__init__()
        self.net = net
        self.x = x
        self.y = y
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def get_parameters(self, config: dict[str, Scalar]) -> list[np.ndarray]:
        return self.net.get_weights()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[list[np.ndarray], int, dict[str, Scalar]]:
        self.net.set_weights(parameters)
        epochs = int(config.get("local-epochs", self.epochs))
        loss = fl_task.train(
            self.net,
            self.x,
            self.y,
            epochs=epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        return self.net.get_weights(), len(self.x), {"train_loss": loss}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        self.net.set_weights(parameters)
        metrics = fl_task.evaluate(self.net, self.x, self.y)
        return float(metrics["loss"]), len(self.x), metrics


def client_fn(context: Context) -> Client:
    """Build the client bound to this supernode's insurance identity."""
    rc = context.run_config
    client_id = fl_task.CLIENT_IDS[int(context.node_id) % len(fl_task.CLIENT_IDS)]
    data_dir = Path(str(rc["data-dir"]))  # run_config values are Scalars
    x, y = fl_task.load_partition(data_dir, client_id, split="train")
    net = fl_task.build_model(input_dim=int(x.shape[1]))
    log.info(
        "node %s -> client %s (%d train rows, %d features)",
        context.node_id,
        client_id,
        len(x),
        x.shape[1],
    )
    return InsuranceNNClient(
        net,
        x,
        y,
        epochs=int(rc["local-epochs"]),
        batch_size=int(rc["batch-size"]),
        lr=float(rc["learning-rate"]),
    ).to_client()


app = ClientApp(client_fn)
