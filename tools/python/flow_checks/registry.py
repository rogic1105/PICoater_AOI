"""Validator registry used by the all-flow entry point."""

from .contract import GlobalContractValidator
from .capture import CaptureFlowValidator
from .data import DataFlowValidator
from .hardware import HardwareFlowValidator
from .live import LiveFlowValidator
from .mura import MuraFlowValidator
from .parameter import ParameterFlowValidator
from .review import ReviewFlowValidator
from .settings import SettingsFlowValidator


VALIDATORS = (
    GlobalContractValidator(),
    LiveFlowValidator(),
    ReviewFlowValidator(),
    DataFlowValidator(),
    CaptureFlowValidator(),
    HardwareFlowValidator(),
    SettingsFlowValidator(),
    MuraFlowValidator(),
    ParameterFlowValidator(),
)

# Keep this explicit so newly documented domains cannot silently appear covered.
PENDING_DOMAINS = ()
