import torch
from torch import nn
from torchvision.transforms import v2

from anomalib.models.image.patchcore.torch_model import PatchcoreModel

class SimplePatchcore(nn.Module):
    """Examplary Model class for track 1.
    This class contains the torch model and the transformation pipeline.
    Forward-pass should first transform the input batch and then pass it through the model.

    Note:
        This is the example model class name. You can replace it with your model class name.

    Args:
        backbone (str): Name of the backbone model to use.
            Default: "wide_resnet50_2".
        layers (list[str]): List of layer names to use.
            Default: ["layer1", "layer2", "layer3"].
        pre_trained (bool): If True, use pre-trained weights.
            Default: True.
        num_neighbors (int): Number of neighbors to use.
            Default: 9.
    """
    def __init__(
        self,
        backbone: str = "mobilenetv3_large_100",
        layers: list[str] = ['blocks.2.2', 'blocks.4.1', 'blocks.6.0'],  # noqa: B006
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()

        # NOTE: Create your transformation pipeline here.
        self.transform = v2.Compose(
            [
                v2.Resize((256, 256)),
                v2.CenterCrop((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
            ],
        )

        self.model = PatchcoreModel(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )
        self.model.eval()

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.model.load_state_dict(state_dict, *args, **kwargs)
        return

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transform the input batch and pass it through the model.
        This model returns a dictionary with the following keys
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """
        batch = self.transform(batch)
        out = self.model(batch)
        # if hasattr(out, "anomaly_map"):
        #     print("[DEBUG] apply sigmoid to anomaly_map")
        #     out["anomaly_map"] = torch.sigmoid(out["anomaly_map"])
        return out
