<div align="center">

# Meta Flow Maps

📄 Paper: [arXiv: 2601.14430](https://arxiv.org/abs/2601.14430) · 🌐 Website: [meta-flow-maps.github.io](https://meta-flow-maps.github.io)

</div>


<div align="center">
  <img src="assets/images/MFMdiagram.png" alt="MFM Diagram" width="800"/>
</div>

### 1. Download Pretrained Models

**MFM XL/2 (ESD-Teacher) Checkpoint**

Available from [Hugging Face Hub](https://huggingface.co/adh1s/mfm):

```bash
pip install huggingface_hub
hf download adh1s/mfm --include "mfm-xl2.pt" --local-dir ckpts
```

By default, all scripts search for the checkpoint at `ckpts/mfm-xl2.pt`. If stored elsewhere, update the corresponding `.yaml` file or override using Hydra syntax.

**DMF XL/2+ Weights** (required for reproducing ImageNet MFM training results)

```bash
hf download kyungmnlee/DMF --local-dir ckpts
```

Set `dmf_path` in `conf/config_train.yaml` or override during training for initialization.

---

### 2. Environment Setup

- **Python**: 3.12
- **GPU Architecture**:
  - Hopper GPUs (H100, H200, H800): CUDA 12.9 + Flash Attention v3
  - Ampere GPUs (A100): Flash Attention v2
- **Flash Attention v3**: Install from source via the [official repository](https://github.com/Dao-AILab/flash-attention)

```bash
conda create -n mfm python=3.12 -y
conda activate mfm
pip install -e .
```

---

### 3. Dataset

Currently supporting [ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) experiments. By default, YAML files search for datasets in `mfm/data`. Update the location in the corresponding `.yaml` files or override using Hydra syntax.

**Example**:
```bash
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py ++data_dir=/path/to/imagenet
```

---

### 4. Training

For maximum efficiency, we recommend GLASS distillation from a well-trained flow map (DMF). Scratch training and training from data are also supported via options in `conf/config_train.yaml`.

**Training Script**: `scripts/train.py`

---

### 5. Evaluation

Evaluate MFM checkpoints using samples from `scripts/sample.py` or `scripts/sample_posterior.py`. Both generate `samples.npz`, which can be evaluated for FID:

```bash
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz samples.npz
```

**Reference Statistics**: Download [ImageNet reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)

---

### 6. Additional Scripts

- **Value Function Estimation**: `scripts/sample_value.py`
- **Inference-Time Steering (MFM-G)**: `scripts/sample_steered.py`
- **MFM-Search**: `scripts/sample_search.py`
- **Fine-Tuning**: `scripts/finetune.py`

---

### 7. Example: Inference-Time Steering (MFM-G)

```bash
torchrun --nnodes=1 --nproc_per_node=1 scripts/sample_steered.py \
  ++drift_estimator=iwae \
  ++mc_samples=16 \
  ++image_reward.prompt="A high-resolution, high-quality photograph of a tabby cat." \
  ++class_label=281
```

---

### Citation

If you use this code or models in your research, please considering citing:

```bibtex
@article{potaptchik2026meta,
  title={Meta Flow Maps enable scalable reward alignment},
  author={Potaptchik, Peter and Saravanan, Adhi and Mammadov, Abbas and Prat, Alvaro and Albergo, Michael S and Teh, Yee Whye},
  journal={arXiv preprint arXiv:2601.14430},
  year={2026}
}
```

---

### Note

If you encounter any difficulties in reproducing our findings, please do let us know.

---

### Acknowledgement

This code borrows model definitions and weights from [DMF](https://github.com/kyungmnlee/dmf). The FID code in `/evaluations` is borrowed from [guided-diffusion](https://github.com/openai/guided-diffusion).
