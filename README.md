# anomaly_detection-MVTEC

# Dataset

[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad)

# Environment & Libs
Reference
[Anomalib](https://github.com/openvinotoolkit/anomalib)

Installing anomalib from source

```bash
# Use of virtual environment is highly recommended
# Using conda
conda create -n anomalib_env python=3.11 -y
conda activate anomalib_env

pip install torch torchvision
pip install tqdm click
pip install einops FrEIA timm open_clip_torch imgaug lightning
pip install kornia
pip install pandas
pip install scikit-learn

pip install anomalib==1.0.1
anomalib install

# pip install -U torchvision einops FrEIA timm open_clip_torch imgaug lightning kornia openvino git+https://github.com/openvinotoolkit/anomalib.git

# pip install -U git+https://github.com/voxel51/fiftyone.git

# pip install -U huggingface_hub umap-learn git+https://github.com/openai/CLIP.git
```

# Tips from the Q&A forum of the competition

"Each included perturbation aims to simulate a type of domain shift that may occur over time in the real world. Examples are Gaussian noise to simulate camera noise or low-light conditions, brightness changes to mimic different times of day, and rotations or translations to simulate varying camera positions. The perturbations will be applied in single or combinations and introduced randomly to ensure variability."

# TODO:
- [x] env. setup.
- [x] Setup simple train and test code
- [x] W&B integration
- [x] Setup the inference script as required by the challenge *
- [x] Basic evaluation.py working as required
- [x] Make a initial submission with the baseline model
- [x] Check if model is actually learning (check prediction and double check eval script)
- [x] Create a dataloader browser to check augmentations
- [] Improve the wandb logs for better benchmark between experiments
- [] Improve launch_train.py to use multiprocess to train multiple models in parallel
- [x] EDA
- [] Test different models
- [x] Explore data augmentations (single or combinations)
    - [x] Low-light condition
    - [x] Random Noise
    - [x] Camera noise
    - [x] Brightness changes
    - [x] Rotations
    - [x] Translations
- [x] Compute evaluation metrics used for benchmarking on the competition