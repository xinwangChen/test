# EDT: An Efficient Diffusion Transformer Framework Inspired by Human-like Sketching

### [Paper](url-ling) | [HuggingFace space](https://huggingface.co/trenkey/EDT)

<!-- ### [Paper](url-ling) | [![HuggingFace space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/trenkey/EDT) -->

<!-- ## MDTv2: Faster Convergeence & Stronger performance
**MDTv2 achieves superior image synthesis performance, e.g., a new SOTA FID score of 1.58 on the ImageNet dataset, and has more than 10× faster learning speed than the previous SOTA DiT.** -->

<!-- MDTv2 demonstrates a 5x acceleration compared to the original MDT.

[MDTv1 code](https://github.com/sail-sg/MDT/tree/mdtv1)
## Introduction

Despite its success in image synthesis, we observe that diffusion probabilistic models (DPMs) often lack contextual reasoning ability to learn the relations among object parts in an image, leading to a slow learning process. To solve this issue, we propose a Masked Diffusion Transformer (MDT) that introduces a mask latent modeling scheme to explicitly enhance the DPMs’ ability to contextual relation learning among object semantic parts in an image. 

During training, MDT operates in the latent space to mask certain tokens. Then, an asymmetric diffusion transformer is designed to predict masked tokens from unmasked ones while maintaining the diffusion generation process. Our MDT can reconstruct the full information of an image from its incomplete contextual input, thus enabling it to learn the associated relations among image tokens. We further improve MDT with a more efficient macro network structure and training strategy, named MDTv2. 

Experimental results show that MDTv2 achieves superior image synthesis performance, e.g., **a new SOTA FID score of 1.58 on the ImageNet dataset, and has more than 10× faster learning speed than the previous SOTA DiT**. 

<img width="800" alt="image" src="figures/vis.jpg"> -->

# Introduction

Transformer-based Diffusion Probabilistic Models (DPMs) have shown more potential than CNN-based DPMs, yet their extensive computational requirements hinder widespread practical applications. To reduce the computation budget of transformer-based DPMs, this work proposes the **E**fficient **D**iffusion **T**ransformer (EDT) framework. The framework includes a lightweight-design diffusion model architecture, and a training-free Attention Modulation Matrix and its alternation arrangement in EDT inspired by human-like sketching. Additionally, we propose a token relation-enhanced masking training strategy tailored explicitly for EDT to augment its token relation learning capability.Our extensive experiments demonstrate the efficacy of EDT. The EDT framework reduces training and inference costs and surpasses existing transformer-based diffusion models in image synthesis performance, thereby achieving a significant overall enhancement. EDT achieved lower FID, EDT-S, EDT-B, and EDT-XL attained speed-ups of 3.93x, 2.84x, and 1.92x respectively in the training phase, and 2.29x, 2.29x, and 2.22x respectively in inference, compared to the corresponding sizes of MDTv2.

<img width="800" alt="image" src="visualization.jpg">

# Performance


| Model| Dataset  | Resolution | Cost(Iter. × BS) | GFLOPs | FID-50K | Inception Score | Weight |
| ------ | ------ | ------ | ------ | ------ | ------- | ------ | ------ |
| EDT-S/2 | ImageNet | 256x256    | 400k × 256       | 2.66   | 34.27   | 42.6  | [google](https://drive.google.com/file/d/1DkglqB4wxlHeDUkerk1G8KqaNcwA_oD-/view?usp=drive_link)([baidu](https://pan.baidu.com/s/1s856mTUODjg6TcsDdMptwQ?pwd=gv0h)) |
| EDT-B/2 | ImageNet | 256x256   | 400k × 256   | 10.20  | 19.18   | 74.4 | [google](https://drive.google.com/file/d/1Zd2bx8JkRKOdRPFpY6PeOQY9zNcf_Fqv/view?usp=drive_link)([baidu](https://pan.baidu.com/s/1EOBbcYrfk7oQfieUf68GgQ?pwd=8e33)) |
| EDT-B/2 | ImageNet | 256x256   | 1000k × 256   | 10.20  | 13.58   | 94.1 | [google](https://drive.google.com/file/d/1UDxgFqoEwGnLZMO__u-BdqzzZ_SolBTc/view?usp=drive_link)([baidu](https://pan.baidu.com/s/1FEeQal8kkabRVi3rSi4fNQ?pwd=6vc0)) |
| EDT-XL/2 | ImageNet | 256x256  | 400k × 256  | 51.83  | 7.52    | 142.4 | [google](https://drive.google.com/file/d/1h583ejF6EUa31f7p34iSpBEjjDpdi5gC/view?usp=drive_link)([baidu](https://pan.baidu.com/s/1E0IAIEkhBQxUNb717iJicg?pwd=rzgn)) |
| EDT-XL/2-G | ImageNet | 256x256   | 2000k × 256  | 51.83  | 3.54  | 355.8 | [google](https://drive.google.com/file/d/1hEZ7IrCuw9OWH0w_r5f_e8mkVZesC5Dj/view?usp=drive_link)([baidu](https://pan.baidu.com/s/1jXbNwDI1Qyr5JCaunrVERQ?pwd=dkac ))|

More model weight([google](https://drive.google.com/drive/folders/1YsXs6NBdCQHQOsD43ijbzukEPTVt6ZeV?usp=drive_link)([baidu](https://pan.baidu.com/s/1N8j-lW3k5T-15JORFiqdmw?pwd=qh1p)))

# Setup

First, download and set up the repo:

```bash
git clone https://github.com/---/EDT.git
cd EDT
```

And then download the pre-trained [VAE-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema) or [VAE-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse):

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate EDT
```

# Evaluation

The evaluation code is obtained from [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations).
Please follow the instructions in the `evaluations` folder to set up the evaluation environment.

[`sample_ddp.py`](sample_ddp.py) script samples a large number of images from a EDT model in parallel. This script
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from pre-trained EDT-S model over `N` GPUs, run:

### Class-conditional sampling:
 
```shell

torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model EDT-S/2 --amm True --sample-dir /path/save/samples --num-fid-samples 50000 --ckpt /path/save/checkpoint.pt

python evaluator.py /path/dataeval/VIRTUAL_imagenet256_labeled.npz /path/save/samples/samples_50000x256x256x3.npz
```

### CFG Class-conditional sampling:

```shell
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model EDT-S/2 --amm True --sample-dir /path/save/samples --cfg-scale 2 --num-fid-samples 50000 --ckpt /path/save/checkpoint.pt

python evaluator.py /path/dataeval/VIRTUAL_imagenet256_labeled.npz /path/save/samples/samples_50000x256x256x3.npz
```

# Visualization

Run the `inferance_edt.py` to generate images.

# Training

### Preparation Before Training

To extract ImageNet features with `1` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --data-path /path/to/imagenet/train --features-path /path/to/store/features
```

### Install Adan optimizer

Install [Adan optimizer](https://github.com/sail-sg/Adan), Adan is a strong optimizer with faster convergence speed than AdamW. [(paper)](https://arxiv.org/abs/2208.06677)

```
python -m pip install git+https://github.com/sail-sg/Adan.git
```

### Training

```shell
accelerate launch --multi_gpu --num_processes 8 --main_process_port 6001 train_mask_adan.py --results-dir /path/save/checkpoint --model EDT-S/2 --init-lr 1e-3 --feature-path /path/to/store/features --epochs 81
```

<!-- # Citation

```

``` -->

# Acknowledgement

This codebase is built based on the [DiT](https://github.com/facebookresearch/dit) , [ADM](https://github.com/openai/guided-diffusion) and [MDT](https://github.com/sail-sg/MDT). Thanks!
