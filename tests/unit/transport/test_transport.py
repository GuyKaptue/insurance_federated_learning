# Test for transport module

from insurance_fl.transport.transport import transport_outcome_model


def test_transport_import():
    """Verify transport module is importable and functional."""

    assert callable(transport_outcome_model)
