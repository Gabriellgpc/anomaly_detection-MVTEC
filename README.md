# anomaly_detection-MVTEC

# Dataset

[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad)

# Environment & Libs

[Anomalib](https://github.com/openvinotoolkit/anomalib)

# Tips from the Q&A forum of the competition

"Each included perturbation aims to simulate a type of domain shift that may occur over time in the real world. Examples are Gaussian noise to simulate camera noise or low-light conditions, brightness changes to mimic different times of day, and rotations or translations to simulate varying camera positions. The perturbations will be applied in single or combinations and introduced randomly to ensure variability."

# TODO:
- [x] env. setup.
- [x] Setup simple train and test code
- [x] W&B integration
- [x] Setup the inference script as required by the challenge *
- [x] Basic evaluation.py working as required
- [] Make a initial submission with the baseline model
- [] Test different models
- [] Explore data augmentations (single or combinations)
    - [] Low-light condition
    - [] Random Noise
    - [] Camera noise
    - [] Brightness changes
    - [] Rotations
    - [] Translations
- [] Compute evaluation metrics used for benchmarking on the competition
    - "The submissions for Category 1 of the challenge will be ranked by evaluating the models on all 15 categories of the MVTec dataset. The F1Max score (highest F1 score that can be achieved based on the raw anomaly score predictions) will be computed for the test set of each of the categories after applying a set of random transformations to the data to simulate domain drift."
- [] EDA