"""Typed glue between the pack's captioners and mlx-vlm.

mlx-vlm is installed only on Apple Silicon, so these keep the calls to it type-correct there without changing what
is passed. Nothing here imports mlx-vlm at run time.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mlx_vlm.generate.types import ProcessorLike
    from PIL import Image


def chat_prompt(template: Callable[..., Any], processor: Any, config: Any, text: str) -> str:
    """The prompt string mlx_vlm.prompt_utils.apply_chat_template (passed as template) builds for text about one
    image. It is typed as messages or a string; it gives messages only when asked (return_messages=True)."""
    prompt = template(processor, config, text, num_images=1)
    if not isinstance(prompt, str):
        raise TypeError(f"mlx-vlm's chat template gave {type(prompt).__name__}, not a prompt string")
    return prompt


def pil_images(*images: Image.Image) -> list[str]:
    """Images for mlx_vlm.generate, as they are. generate annotates image as str | list[str], but the code it runs
    takes a PIL image unchanged (process_image in mlx_vlm/utils.py converts only a str), so the cast is to its
    annotation, not a conversion: turning each frame into a path or a data URL would encode and decode it again."""
    return cast("list[str]", list(images))


def loaded_processor(processor: object) -> ProcessorLike:
    """The processor mlx_vlm.load returns, as mlx_vlm.generate takes it. load annotates it as transformers'
    ProcessorMixin, but its load_processor (mlx_vlm/utils.py) attaches the detokenizer that generate's ProcessorLike
    needs, so this checks for it and casts: the object is passed on unchanged."""
    if not hasattr(processor, "detokenizer"):
        raise TypeError("mlx-vlm's processor has no detokenizer, so it can't generate text")
    return cast("ProcessorLike", processor)
