"""Sample evaluation script for track 1."""

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn

from anomalib.data import MVTec
from anomalib.metrics import F1Max


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    return parser.parse_args()


def load_model(module_path: str, class_name: str, weights_path: str, category: str) -> nn.Module:
    """Load model.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str): Path to the model weights.
        category (str): Category of the dataset.

    Note:
        We assume that the weight path contain the weights for all categories.
            For example, if the weight path is "/path/to/weights/", then the
            weights for each category should be stored as
            "/path/to/weights/bottle.pth", "/path/to/weights/zipper.pth", etc.


    Returns:
        nn.Module: Loaded model.
    """
    # get model class
    model_class = getattr(importlib.import_module(module_path), class_name)
    # instantiate model
    model = model_class()
    # load weights
    if weights_path:
        weight_file = Path(weights_path) / f"{category}.pth"
        model.load_state_dict(torch.load(weight_file))
    return model


def run(module_path: str, class_name: str, weights_path: str, dataset_path: str, category: str) -> None:
    """Run the evaluation script.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str | None, optional): Path to the model weights.
        dataset_path (str): Path to the dataset.
        category (str): Category of the dataset.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instantiate and load the model
    model = load_model(module_path, class_name, weights_path, category)
    model.to(device)

    # Create the dataset
    # NOTE: We fix the image size to (256, 256) for consistent evaluation across all models.
    datamodule = MVTec(root=dataset_path, eval_batch_size=1, image_size=(256, 256))
    datamodule.setup()

    # Create the metrics
    image_metric = F1Max()
    pixel_metric = F1Max()

    # Loop over the test set and compute the metrics
    for data in datamodule.test_dataloader():
        output = model(data["image"].to(device))

        # Update the image metric
        image_metric.update(output["pred_score"].cpu(), data["label"])

        # Update the pixel metric
        pixel_metric.update(output["anomaly_map"].squeeze().cpu(), data["mask"].squeeze().cpu())

    # Compute the metrics
    image_score = image_metric.compute()
    print(image_score)
    pixel_score = pixel_metric.compute()
    print(pixel_score)


if __name__ == "__main__":
    args = parse_args()
    run(args.module_path, args.class_name, args.weights_path, args.dataset_path, args.category)
