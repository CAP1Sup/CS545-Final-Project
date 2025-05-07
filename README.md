# Student and Teacher - Distillation Applied to the 2nd Edition of FRCSyn

## Overview

This repository contains the code and data for the paper "Student and Teacher - Distillation Applied to the 2nd Edition of FRCSyn". The paper presents attempts to apply distillation in the context of [the 2nd edition of FRCSyn](https://frcsyn.github.io/CVPR2024.html). The goal is to improve the performance of the student model by leveraging the knowledge of a teacher model.

## Requirements

- Linux. The code has been tested on Ubuntu 24.04 LTS
- Python 3.8 or higher. Python 3.12 is recommended
- NVidia GPU with CUDA support (for training the models)
- CUDA 12.2

## Setup

### General Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/CAP1Sup/CS545-Final-Project.git
   cd CS545-Final-Project
   ```

2. Clone the submodules:

   ```bash
   git submodule update --init --recursive
   ```

3. Download the StyleGAN2 model weights. Please visit the [official page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2/files) and download `stylegan2-ffhq-256x256.pkl` model. Place the downloaded file in the main directory of the repository. The directory structure should look like this:

   ```text
   CS545-Final-Project/
   ├── datasets/
   ├── results/
   ├── stylegan2-ffhq-256x256.pkl
   └── ...
   ```

4. Create and activate the virtual environment:

   ```bash
    python3 -m venv venv # Or python3.12 -m venv venv for Python 3.12
    source venv/bin/activate # Always activate the virtual environment before running any scripts
   ```

5. Install the required packages:

   ```bash
   pip3 install -r requirements.txt
   ```

### Evaluation Setup

1. Make a folder for the datasets:

   ```bash
   # Ensure you are in the root directory of the repository (CS545-Final-Project)
   mkdir datasets
   ```

2. Download the aligned versions of the required datasets. All datasets should be placed in a `datasets` folder. You can download them from the following links:

   - [LFW](https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT/view?usp=sharing)
   - [CALFW](https://drive.google.com/file/d/1kpmcDeDmPqUcI5uX0MCBzpP_8oQVojzW/view?usp=sharing)
   - [CPLFW](https://drive.google.com/file/d/14vPvDngGzsc94pQ4nRNfuBTxdv7YVn2Q/view?usp=sharing)
   - [AgeDB](https://drive.google.com/file/d/1AoZrZfym5ZhdTyKSxD0qxa7Xrp2Q1ftp/view?usp=sharing)
   - [CFP FF and FP](https://drive.google.com/file/d/1-sDn79lTegXRNhFuRnIRsgdU88cBfW6V/view?usp=sharing)
   - [VGG2-FP](https://drive.google.com/file/d/1N7QEEQZPJ2s5Hs34urjseFwIoPVSmn4r/view?usp=sharing)

3. Extract the downloaded datasets into the `datasets` folder. The folder structure should look like this:

   ```text
   datasets/
   ├── agedb_30
   ├── calfw
   ├── cfp_ff
   ├── cfp_fp
   ├── cplfw
   ├── lfw
   └── vgg2_fp
   ```

   **Note:** The `datasets` folder should contain `.npy` and `.bin` files in its root directory. Ensure these files are correctly extracted and placed in the main `datasets` folder for evaluation.

## Usage

### Training the Student Models

If you wish to train a model that is already within the results folder, please delete the model's folder from the results folder before running the training command. For example, if you want to train a ResNet50 model, delete the `results/timm/resnet50.a1_in1k` folder before running the training command.

To train a student model, run the following command:

```bash
python3 train.py --model "author/model"                  # Generic example
python3 train.py --model "timm/resnet50.a1_in1k"         # For ResNet50
python3 train.py --model "timm/resnext50_32x4d.a1h_in1k" # For ResNeXt50
```

To generate graphs after the training process has completed, run the following command:

```bash
python3 graph_training.py "author/model"                  # Generic example
python3 graph_training.py "timm/resnet50.a1_in1k"         # For ResNet50
python3 graph_training.py "timm/resnext50_32x4d.a1h_in1k" # For ResNeXt50
```

### Evaluating the Student Models

Edit the `benchmark_config.py` file to set the model for evaluation. For example, if you trained the ResNet50 model and found that the 9th epoch produced the best model, set the model in `benchmark_config.py` to:

```python
BACKBONE_NAME="Custom", # To indicate a custom model
BACKBONE_RESUME_ROOT="./results/timm/resnet50.a1_in1k/epoch_9.pt", # The path to the model's checkpoint
```

If the backbone name is set to `ArcFace`, the teacher model will be used for evaluation. The config's model section would look like this:

```python
BACKBONE_NAME="ArcFace",
```

The rest of the config file should not need to be modified. Then run the following command to evaluate the model:

```bash
python3 benchmark.py
```

## Attribution

This repository builds on work from the [SynthDistill](https://gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill) and [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe) repositories. Please refer to their respective repositories for more information.
