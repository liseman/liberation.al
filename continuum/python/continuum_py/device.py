"""Where a worker computes, and slab views as torch tensors on that device (docs/CONTINUUM.md, "SDKs" and
"Guardrails": every worker picks cuda, mps, or cpu at start, from one setting, so the same code runs on the 3090s
and on the Mac).

torch is imported only when these functions run, so the rest of continuum_py doesn't need it.

- ``pick_device()`` decides once: an explicit setting, else $CONTINUUM_DEVICE, else the best this machine has (cuda,
  then mps, then cpu). A device that was asked for and isn't here raises, rather than quietly running somewhere else.
- ``to_tensor(view, device)`` turns a leased slot (or anything with ``__dlpack__``) into a tensor on that device. On
  the CPU it's the slot itself, through DLPack, with no copy, and it holds the slot's pin until it's collected. On CUDA
  it's a copy by DMA from the arena; on MPS a copy into Metal's memory (torch can't wrap host memory as an MPS
  tensor).
- ``register_arena(arena)`` registers the whole arena with CUDA once per process (``cudaHostRegister``), so copies
  from it are true DMA and can run asynchronously. Registering faults in every page, so the first CUDA worker makes
  the arena resident.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

    from continuum_py._native import Arena

ENV = "CONTINUUM_DEVICE"
"""The setting a worker's device comes from when it's not passed in: cuda, cuda:N, mps, cpu, or auto."""

DeviceKind = Literal["cuda", "mps", "cpu"]

_SETTING = re.compile(r"auto|cpu|mps|cuda(:\d+)?")

# Arenas registered with CUDA in this process, by base address. Holding the Arena keeps its mapping alive while
# it's registered: unmapping registered memory without cudaHostUnregister is undefined for CUDA.
_registered: dict[int, Arena] = {}


class DeviceUnavailable(RuntimeError):
    """The device asked for isn't on this machine (or torch can't use it)."""


def pick_device(setting: str | None = None) -> str:
    """The device to compute on, as torch spells it ("cuda", "cuda:1", "mps", or "cpu").

    ``setting`` wins, then $CONTINUUM_DEVICE, then "auto": cuda if torch sees a GPU, else mps, else cpu. Raises
    ValueError for a setting that isn't one of those, and DeviceUnavailable for a device this machine doesn't have.
    """
    import torch

    chosen = (setting or os.environ.get(ENV) or "auto").strip().lower()
    if not _SETTING.fullmatch(chosen):
        raise ValueError(f"device must be auto, cpu, mps, cuda, or cuda:N, not {chosen!r}")
    if chosen == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if chosen.startswith("cuda"):
        if not torch.cuda.is_available():
            raise DeviceUnavailable(f"{chosen} was asked for, but torch sees no CUDA device here")
        index = int(chosen.split(":")[1]) if ":" in chosen else None
        if index is not None:
            if index >= torch.cuda.device_count():
                raise DeviceUnavailable(
                    f"{chosen} was asked for, but there are {torch.cuda.device_count()} CUDA devices"
                )
            return f"cuda:{index}"  # "cuda:01" becomes "cuda:1"
    if chosen == "mps" and not torch.backends.mps.is_available():
        raise DeviceUnavailable("mps was asked for, but torch can't use Metal here")
    return chosen


def to_tensor(
    view: Any,
    device: str = "cpu",
    *,
    dtype: torch.dtype | None = None,
    shape: list[int] | tuple[int, ...] | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    """A leased slot (or anything with ``__dlpack__``) as a torch tensor on ``device``.

    The slot's bytes are reinterpreted as ``dtype`` (default uint8) and reshaped to ``shape`` (default flat). On the
    CPU this is the slot itself, read-only by contract, with no copy; the tensor holds the slot's pin until it's
    collected. On CUDA and MPS it's a copy, after which the slot is no longer needed.

    ``non_blocking=True`` lets a CUDA copy run asynchronously, which only happens when the arena is registered
    (``register_arena``). The slot has to stay pinned until the copy is done, so the returned tensor keeps the
    source view (and with it the pin) for as long as it lives; release the pin sooner by synchronizing the stream
    and dropping the GPU tensor, or copy without non_blocking.
    """
    import torch

    flat = torch.from_dlpack(view)
    if dtype is not None and dtype != flat.dtype:
        flat = flat.view(dtype)
    tensor = flat.reshape(shape) if shape is not None else flat
    if torch.device(device).type == "cpu":
        return tensor
    out = tensor.to(device, non_blocking=non_blocking)
    if non_blocking and out.data_ptr() != tensor.data_ptr():
        # The copy may still be reading the slot when this returns: keep the source (and its pin) with the result.
        out._continuum_source = tensor  # ty: ignore[unresolved-attribute]
    return out


def _cudart() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise DeviceUnavailable("registering an arena with CUDA needs a CUDA device, and torch sees none here")
    return torch.cuda.cudart()


def register_arena(arena: Arena) -> None:
    """Registers the whole arena with CUDA in this process (``cudaHostRegister``, default flags), so copies from any
    slot are DMA and ``to_tensor(..., non_blocking=True)`` can overlap them with compute. Faults in every page.

    Registering an arena that's already registered raises ValueError before CUDA sees it (CUDA would refuse it and
    leave an error that fails the next unrelated kernel launch). The arena stays mapped until ``unregister_arena``.
    Raises DeviceUnavailable without CUDA, and RuntimeError (with CUDA's message) if registration fails.
    """
    cudart = _cudart()
    if arena.base_address in _registered:
        raise ValueError(f"arena {arena.name} is already registered with CUDA in this process")
    rc = cudart.cudaHostRegister(arena.base_address, arena.size, 0)
    if int(rc) != 0:
        raise RuntimeError(f"cudaHostRegister failed for arena {arena.name}: {cudart.cudaGetErrorString(rc)}")
    _registered[arena.base_address] = arena


def unregister_arena(arena: Arena) -> None:
    """Undoes ``register_arena`` (``cudaHostUnregister``). ValueError if the arena isn't registered."""
    cudart = _cudart()
    if _registered.pop(arena.base_address, None) is None:
        raise ValueError(f"arena {arena.name} isn't registered with CUDA in this process")
    rc = cudart.cudaHostUnregister(arena.base_address)
    if int(rc) != 0:
        raise RuntimeError(f"cudaHostUnregister failed for arena {arena.name}: {cudart.cudaGetErrorString(rc)}")
