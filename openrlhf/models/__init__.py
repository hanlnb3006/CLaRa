#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

_ACTOR_IMPORT_ERROR = None

try:
    from .actor import Actor
except ModuleNotFoundError as exc:
    if exc.name not in {"flash_attn", "ring_flash_attn"}:
        raise
    _ACTOR_IMPORT_ERROR = exc

from .loss import (
    DPOLoss,
    GPTLMLoss,
    KDLoss,
    KTOLoss,
    LogExpLoss,
    PairWiseLoss,
    PolicyLoss,
    PRMLoss,
    SFTLoss,
    ValueLoss,
    VanillaKTOLoss,
)

__all__ = [
    "SFTLoss",
    "DPOLoss",
    "GPTLMLoss",
    "KDLoss",
    "KTOLoss",
    "LogExpLoss",
    "PairWiseLoss",
    "PolicyLoss",
    "PRMLoss",
    "ValueLoss",
    "VanillaKTOLoss",
]

if _ACTOR_IMPORT_ERROR is None:
    __all__.insert(0, "Actor")


def __getattr__(name):
    if name == "Actor" and _ACTOR_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Actor requires optional training dependencies flash_attn/ring_flash_attn. "
            "Install them for training paths, or import CLaRa directly for inference/evaluation."
        ) from _ACTOR_IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
