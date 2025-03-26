# Inter Sample Token Alignment Attention (ISTA2)

## Introduction

This repository provides the official implementation of "ISTA2: Inter-Sample Token Alignment for Vision Transformers", supporting training and evaluation on ImageNet-1K image classification task for three baseline architectures: PVTv2, Swin Transformer, and CvT. The code has been anonymised for review purposes.

### Abstract

<details>
    <summary><i><b>Click here</b></i></summary>
    Attention mechanisms have emerged as a fundamental component across vision and language models; they can be broadly categorized into self-attention (intra-sequence relationships) and cross-attention (inter-sequence alignment). While cross-attention traditionally bridges heterogeneous data like cross-lingual or cross-modal tasks, we extend its capability to homogeneous image data through a novel Inter-Sample Token Alignment Attention (ISTA2) module. ISTA2 introduces a new nested attention architecture that integrates cross-sample token alignment through a value-based cross-attention module in place of standard self-attention. To address the quadratic complexity inherent in pairwise cross-sample token comparisons, we developed a parameter-efficient representative token selection scheme. The proposed approaches enhance transformers' capacity on modelling inter-sample dependencies while further facilitating semantic alignment of similar tokens across instances. Evaluations on ImageNet-1K for the classification task demonstrate ISTA2's effectiveness, yielding consistent top-1 accuracy improvements of 0.5-1.1% across three popular baseline architectures without significant computational overhead.
</details>

### Main Results

|     Method     | Resolution | Acc@1 | #Param (M) | GFLOPs |                            Config                            |
| :------------: | :--------: | :---: | :--------: | :----: | :----------------------------------------------------------: |
|    PVTv2-B0    |  224x224   | 70.8  |    3.7     |  0.6   | [config](PVT/classification/configs/pvt_v2_ista2/pvt_v2_b0.py) |
| PVTv2-B0+ISTA2 |  224x224   | 71.7  |    3.7     |  0.6   | [config](PVT/classification/configs/pvt_v2_ista2/pvt_v2_b0_ista2.py) |
|    PVTv2-B2    |  224x224   | 81.9  |    25.4    |  4.0   | [config](PVT/classification/configs/pvt_v2_ista2/pvt_v2_b2.py) |
| PVTv2-B2+ISTA2 |  224x224   | 82.4  |    25.4    |  4.1   | [config](PVT/classification/configs/pvt_v2_ista2/pvt_v2_b2_ista2.py) |
|     Swin-T     |  224x224   | 81.2  |    28.3    |  4.5   | [config](SWIN/classification/configs/swin_ista2/swin_tiny_patch4_window7_224.yaml) |
|  Swin-T+ISTA2  |  224x224   | 82.3  |    28.3    |  4.9   | [config](SWIN/classification/configs/swin_ista2/swin_tiny_ista2_patch4_window7_224.yaml) |
|     CvT-13     |  224x224   | 81.6  |    20.0    |  4.6   | [config](CVT/classification/experiments/imagenet/cvt_ista2/cvt-13-224x224.yaml) |
|  CvT-13+ISTA2  |  224x224   | 82.3  |    20.0    |  4.7   | [config](CVT/classification/experiments/imagenet/cvt_ista2/cvt-13-ista2-224x224.yaml) |

### Citation

If you find this work useful, please cite:

```bibtex
TBC
```

## Quick Start

### Installation

#### Prerequisites

- Linux-based OS (tested on Ubuntu 22.04)
- [Conda](https://www.anaconda.com/docs/main)
- Python 3.11
- NVIDIA GPU with CUDA 11.8 or 12.1

#### Environment Setup

1. Create a Conda environment:

   ```bash
   conda create -n ista2 python=3.11 -y
   conda activate ista2
   ```

2. Install PyTorch 2.4.0:

   ```bash
   # CUDA 11.8
   pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12.1
   pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install dependencies:

   ```bash
   # General requirements
   pip install numpy==1.26.4 pyyaml==6.0.1 scipy==1.15.2 timm==1.0.11 yacs==0.1.8
   
   # Requirements for PVTv2
   pip install mmcv==2.2.0
   
   # Requirements for Swin Transformer
   pip install opencv-python==4.11.0.86 termcolor==2.5.0
   
   # Requirements for CvT
   pip install einops==0.8.0 json_tricks==3.17.3 pandas==2.2.3 ptflops==0.7.4 \
       scikit-learn==1.6.1 tensorboardX==2.6.2.2 tensorwatch==0.9.1
   ```

#### Clone Repository

```bash
git clone https://anonymous.4open.science/r/Inter-Sample-Token-Alignment.git # To be replaced with the original URL
cd Inter-Sample-Token-Alignment
```

### Data Preparation

This project uses the standard [ImageNet-1K](https://www.image-net.org/) dataset for both training and evaluation. Download and organise the dataset as follows:

```plaintext
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_18.JPEG
│   │   ├── n01440764_36.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   ├── n01443537_2.JPEG
│   │   ├── n01443537_16.JPEG
│   │   └── ...
│   └── ... 
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   └── ...
    ├── n01443537/
    │   ├── ILSVRC2012_val_00000236.JPEG
    │   ├── ILSVRC2012_val_00000262.JPEG
    │   └── ...
    └── ...
```

Set `<imagenet-path>` to your dataset path (e.g., '/data/imagenet') in the subsequent commands.

### Execution

For path/directory placeholders such as `<imagenet-path>`, `<config-file-path>`, `<checkpoint-path>`, and `<output-directory>`, replace them with your actual paths/directories and:

1. **Use absolute path** to avoid potential path resolution errors, **or**
2. **Adjust relative path** according to your current working directory when executing commands.

#### Training

To train a model on ImageNet-1K on a single node for 300 epochs:

- PVTv2

  ```bash
  cd PVT/classification
  torchrun --nproc_per_node <num-of-gpus-to-use> --master_port 11000 main.py \
      --config <config-file-path> --data-path <imagenet-path> \
      [--output_dir <output-directory> --batch-size <batch-size-per-gpu> ...]
  ```

  **Note**: The options in square brackets are optional. For additional options, refer to the [main.py](PVT/classification/main.py) file for PVTv2.

- Swin Transformer

  ```bash
  cd SWIN/classification
  torchrun --nproc_per_node <num-of-gpus-to-use> --master_port 12000 main.py \
      --cfg <config-file-path> --data-path <imagenet-path> \
      [--output <output-directory> --batch-size <batch-size-per-gpu> ...]
  ```

  **Note**: For additional options, refer to the [main.py](SWIN/classification/main.py) file for Swin Transformer and the [get_started.md](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) in the original Swin Transformer repository.

- CvT

  ```bash
  cd CVT/classification
  bash run.sh -g <num-of-gpus-to-use> -t train -p 13000 \
      --cfg <config-file-path> DATASET.ROOT <imagenet-path> \
      [OUTPUT_DIR <output-directory> TRAIN.BATCH_SIZE_PER_GPU <batch-size-per-gpu> ...]
  ```
  
  **Note**: Items in square brackets represent optional KEY-VALUE pairs for configuration. For additional options, refer to [default.py](CVT/classification/lib/config/default.py). To explore further usages of `run.sh`, refer to the [README.md](https://github.com/microsoft/CvT/blob/main/README.md) file in the original CvT repository.

#### Evaluation

To evaluate a pre-trained model on ImageNet-1K validation set with a single node:

- PVTv2

  ```bash
  cd PVT/classification
  # For single GPU evaluation
  torchrun --nproc_per_node 1 --master_port 11000 main.py --eval \
      --config <config-file-path> --resume <checkpoint-path> --data-path <imagenet-path> \
      [--batch-size <batch-size-per-gpu> ...]
  # For multi-GPU evaluation
  torchrun --nproc_per_node <num-of-gpus-to-use> --master_port 11000 main.py --eval --dist-eval \
      --config <config-file-path> --resume <checkpoint-path> --data-path <imagenet-path> \
      [--batch-size <batch-size-per-gpu> ...]
  ```

- Swin Transformer

  ```bash
  cd SWIN/classification
  torchrun --nproc_per_node <num-of-gpus-to-use> --master_port 12000 main.py --eval \
      --cfg <config-file-path> --resume <checkpoint-path> --data-path <imagenet-path> \
      [--batch-size <batch-size-per-gpu> ...]
  ```

- CvT

  ```bash
  cd CVT/classification
  bash run.sh -g <num-of-gpus-to-use> -t test -p 13000 --cfg <config-file-path> \
      TEST.MODEL_FILE <checkpoint-path> DATASET.ROOT <imagenet-path> \
      [TEST.BATCH_SIZE_PER_GPU <batch-size-per-gpu> ...]
  ```

## License

This repository is released under the MIT License. Please see the [LICENSE](LICENSE) file for more information.
