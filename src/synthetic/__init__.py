# src/synthetic/__init__.py
""" Initialization file for the synthetic package. """

from .data_generator import EuropeanInsuranceFederatedDataset as DataGenerator

__all__ = ["DataGenerator"]