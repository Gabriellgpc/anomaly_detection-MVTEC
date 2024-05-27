"""Example model file for track 2."""

import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import resize

from anomalib.models.image.winclip.torch_model import WinClipModel


class MyModel(nn.Module):
    """Example model class for track 2.

    This class applies few-shot anomaly detection using the WinClip model from Anomalib.
    """

    def __init__(self) -> None:
        super().__init__()

        # NOTE: Create your transformation pipeline (if needed).
        self.transform = v2.Compose(
            [
                v2.Resize((240, 240)),
                v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )

        # NOTE: Create your model.
        self.model = WinClipModel()
        self.model.eval()

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transform the input batch and pass it through the model.

        This model returns a dictionary with the following keys
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """
        batch = self.transform(batch)
        pred_score, anomaly_maps = self.model(batch)
        # resize back to 256x256 for evaluation
        anomaly_maps = resize(anomaly_maps, (256, 256))
        return {"pred_score": pred_score, "anomaly_map": anomaly_maps}

    def setup(self, data: dict) -> None:
        """Setup the few-shot samples for the model.

        The evaluation script will call this method to pass the k images for few shot learning and the object class
        name. In the case of MVTec LOCO this will be the dataset category name (e.g. breakfast_box). Please contact
        the organizing committee if if your model requires any additional dataset-related information at setup-time.
        """
        few_shot_samples = data.get("few_shot_samples")
        class_name = data.get("dataset_category")

        few_shot_samples = self.transform(few_shot_samples)
        self.model.setup(class_name, few_shot_samples)
