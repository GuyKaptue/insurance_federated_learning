"""Flower 1.31 gRPC-RERE node-auth behavior, exercised without network access."""

from __future__ import annotations

from types import SimpleNamespace

import grpc
import pytest

pytest.importorskip("flwr")

from flwr.common import now
from flwr.common.constant import PUBLIC_KEY_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER
from flwr.server.superlink.fleet.grpc_rere.node_auth_server_interceptor import (
    NodeAuthServerInterceptor,
)
from flwr.supercore.primitives.asymmetric import (
    generate_key_pairs,
    public_key_to_bytes,
    sign_message,
)


class _State:
    def __init__(self, public_key: bytes, node_id: int) -> None:
        self.public_key = public_key
        self.node_id = node_id

    def get_node_id_by_public_key(self, public_key: bytes) -> int | None:
        return self.node_id if public_key == self.public_key else None


class _StateFactory:
    def __init__(self, state: _State) -> None:
        self._state = state

    def state(self) -> _State:
        return self._state


class _Abort(Exception):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details


class _Context:
    def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise _Abort(code, details)


@pytest.mark.parametrize("requested_node_id,authorized", [(41, True), (99, False)])
def test_flower_node_auth_binds_signed_key_to_registered_node(
    requested_node_id: int, authorized: bool
) -> None:
    private_key, public_key = generate_key_pairs()
    public_bytes = public_key_to_bytes(public_key)
    timestamp = now().isoformat()
    metadata = [
        (PUBLIC_KEY_HEADER, public_bytes),
        (TIMESTAMP_HEADER, timestamp),
        (SIGNATURE_HEADER, sign_message(private_key, timestamp.encode("ascii"))),
    ]
    interceptor = NodeAuthServerInterceptor(
        _StateFactory(_State(public_bytes, node_id=41))  # type: ignore[arg-type]
    )
    details = SimpleNamespace(
        method="/flwr.proto.Fleet/SendNodeHeartbeat",
        invocation_metadata=metadata,
    )

    def continuation(_details):
        return grpc.unary_unary_rpc_method_handler(lambda request, _context: request.node_id)

    handler = interceptor.intercept_service(continuation, details)

    if authorized:
        assert (
            handler.unary_unary(SimpleNamespace(node_id=requested_node_id), _Context())
            == requested_node_id
        )
    else:
        with pytest.raises(_Abort) as exc:
            handler.unary_unary(SimpleNamespace(node_id=requested_node_id), _Context())
        assert exc.value.code == grpc.StatusCode.UNAUTHENTICATED
        assert exc.value.details == "Invalid node ID"


def test_secure_channel_configuration_failure_does_not_fall_back_to_plaintext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flwr.common.grpc import create_channel

    def fail_tls_credentials(_root_certificates):
        raise RuntimeError("synthetic TLS credential failure")

    def fail_if_plaintext_is_attempted(*_args, **_kwargs):
        pytest.fail("TLS setup must not retry with an insecure channel")

    monkeypatch.setattr(grpc, "ssl_channel_credentials", fail_tls_credentials)
    monkeypatch.setattr(grpc, "insecure_channel", fail_if_plaintext_is_attempted)

    with pytest.raises(ValueError, match="Failed to create SSL channel credentials"):
        create_channel("localhost:1234", insecure=False, root_certificates=b"test-root-certificate")
