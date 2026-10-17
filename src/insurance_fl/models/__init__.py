"""
insurance_fl.models — ML models for the Insurance Federated Learning Platform.

Includes multi-task neural network, tree ensemble, and hybrid model
implementations with get_weights/set_weights interface for federated learning.
"""

from insurance_fl.models.multi_task_nn import MultiTaskNN

__all__ = ["MultiTaskNN"]