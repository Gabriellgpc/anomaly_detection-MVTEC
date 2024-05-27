"""Sample evaluation script for track 2."""

import argparse
import importlib
import importlib.util

import torch
from torch import nn

# NOTE: The following MVTecLoco import is not available in anomalib v1.0.1.
# It will be available in v1.1.0 which will be released on April 29th, 2024.
# If you are using an earlier version of anomalib, you could install anomalib
# from the anomalib source code from the following branch:
# https://github.com/openvinotoolkit/anomalib/tree/feature/mvtec-loco
from anomalib.data import MVTecLoco
from anomalib.metrics.f1_max import F1Max

FEW_SHOT_SAMPLES = [1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=False)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    return parser.parse_args()


def load_model(module_path: str, class_name: str, weights_path: str) -> nn.Module:
    """Load model.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str): Path to the model weights.

    Returns:
        nn.Module: Loaded model.
    """
    # get model class
    model_class = getattr(importlib.import_module(module_path), class_name)
    # instantiate model
    model = model_class()
    # load weights
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model


def run(module_path: str, class_name: str, weights_path: str, dataset_path: str, category: str) -> None:
    """Run the evaluation script.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str): Path to the model weights.
        dataset_path (str): Path to the dataset.
        category (str): Category of the dataset.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instantiate model class here
    # Load the model here from checkpoint.
    model = load_model(module_path, class_name, weights_path)
    model.to(device)

    #
    # Create the dataset
    # NOTE: We fix the image size to (256, 256) for consistent evaluation across all models.
    datamodule = MVTecLoco(root=dataset_path, eval_batch_size=1, image_size=(256, 256))
    datamodule.setup()

    #
    # Create the metrics
    image_metric = F1Max()
    pixel_metric = F1Max()

    #
    # pass few-shot images and dataset category to model
    setup_data = {
        "few_shot_samples": torch.stack([datamodule.train_data[idx]["image"] for idx in FEW_SHOT_SAMPLES]).to(device),
        "dataset_category": category,
    }
    model.setup(setup_data)

    # Loop over the test set and compute the metrics
    for data in datamodule.test_dataloader():
        output = model(data["image"].to(device))

        # Update the image metric
        image_metric.update(output["pred_score"].cpu(), data["label"])

        # Update the pixel metric
        if "anomaly_map" in output:
            pixel_metric.update(output["anomaly_map"].squeeze().cpu(), data["mask"].squeeze().cpu())

    # Compute the metrics
    image_score = image_metric.compute()
    print(image_score)
    if pixel_metric.update_called:
        pixel_score = pixel_metric.compute()
        print(pixel_score)


if __name__ == "__main__":
    args = parse_args()
    run(args.module_path, args.class_name, args.weights_path, args.dataset_path, args.category)
