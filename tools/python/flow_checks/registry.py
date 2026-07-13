"""Validator registry used by the all-flow entry point."""

from .contract import GlobalContractValidator
from .data import DataFlowValidator
from .review import ReviewFlowValidator


VALIDATORS = (
    GlobalContractValidator(),
    ReviewFlowValidator(),
    DataFlowValidator(),
)

# These domains remain visible so an all-flow run cannot be mistaken for full coverage.
PENDING_DOMAINS = (
    "LIVE/F",
    "CAPTURE/C",
    "SETTINGS/S",
    "MURA/M",
    "PARAM/P",
    "HARDWARE/H",
)
