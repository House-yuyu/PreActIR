from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from preactir.tools.base import RestorationTool
from preactir.utils.image import image_to_tensor, tensor_to_image


class TorchModuleTool(RestorationTool):
    """Adapter for a pretrained PyTorch restoration model.

    The wrapped model is expected to map a BCHW RGB tensor in [0, 1] to a BCHW
    RGB tensor in [0, 1]. Supply custom preprocessing/postprocessing callables
    for models with different conventions.
    """

    def __init__(
        self,
        name: str,
        targets: tuple[str, ...],
        model: torch.nn.Module,
        device: str | torch.device = "cuda",
        preprocess: Callable[[torch.Tensor, float], torch.Tensor] | None = None,
        postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cost_prior: float = 1.0,
    ) -> None:
        self.name = name
        self.targets = targets
        self.model = model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.cost_prior = float(cost_prior)

    @torch.inference_mode()
    def _apply(self, image: np.ndarray, strength: float) -> np.ndarray:
        tensor = image_to_tensor(image).unsqueeze(0).to(self.device)
        if self.preprocess is not None:
            tensor = self.preprocess(tensor, strength)
        output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, dict):
            for key in ("output", "restored", "image", "pred"):
                if key in output:
                    output = output[key]
                    break
            else:
                raise KeyError("Could not infer image tensor from model output dictionary")
        if self.postprocess is not None:
            output = self.postprocess(output)
        return tensor_to_image(output.clamp(0, 1))
