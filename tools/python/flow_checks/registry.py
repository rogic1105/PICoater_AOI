"""Validator registry used by the all-flow entry point."""

from .contract import GlobalContractValidator
from .capture import CaptureFlowValidator
from .data import DataFlowValidator
from .hardware import HardwareFlowValidator
from .live import LiveFlowValidator
from .review import ReviewFlowValidator


VALIDATORS = (
    GlobalContractValidator(),
    LiveFlowValidator(),
    ReviewFlowValidator(),
    DataFlowValidator(),
    CaptureFlowValidator(),
    HardwareFlowValidator(),
)

# These domains remain visible so an all-flow run cannot be mistaken for full coverage.
PENDING_DOMAINS = (
    "SETTINGS/S",
    "MURA/M",
    "PARAM/P",
)
