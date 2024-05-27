# anomaly_detection-MVTEC

# Dataset

[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad)

# Environment & Libs

[Anomalib](https://github.com/openvinotoolkit/anomalib)


# TODO:
[x] env. setup.
[x] Setup simple train and test code
[x] W&B integration
[] Validation routine | plots and mask visualization
[] Test different models
[] Explore data augmentations
[] Compute evaluation metrics used for benchmarking on the competition
    - "The submissions for Category 1 of the challenge will be ranked by evaluating the models on all 15 categories of the MVTec dataset. The F1Max score (highest F1 score that can be achieved based on the raw anomaly score predictions) will be computed for the test set of each of the categories after applying a set of random transformations to the data to simulate domain drift."
[] EDA
[] EDA with fiftyone