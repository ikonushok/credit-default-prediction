"""Credit-default scoring pipeline (Alfa-Bank x MIPT case).

Predicts P(flag=1) — probability a client defaults — from long-format
credit-product history aggregated to one feature vector per `id`.
"""

__all__ = [
    "config",
    "data_io",
    "features",
    "aggregate",
    "cv",
    "metrics",
    "submission",
]
