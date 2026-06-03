# InfoQuant

Official implementation for **InfoQuant: Reducing Quantization Information Error via Peak Suppression Orthogonal Transformation**.

Paper: https://arxiv.org/abs/2605.26175

InfoQuant optimizes peak-suppression orthogonal transformation matrices for LLaMA-family models, then evaluates quantized models with optional Learning Activation Clipping (LAC).

## Environment

Requirements:

- Python >= 3.9
- PyTorch >= 2.0
- CUDA-capable GPU for actual model rotation, quantization, and evaluation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Fast Hadamard Transform:

```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install .
```

Download LLaMA-2 or LLaMA-3 weights locally from Hugging Face before running experiments.

## Run PSOT

```bash
sh scripts/get_matrix.sh <MODEL_PATH> <BLOCK_NUM> <W_BITS> <A_BITS> <KV_BITS>
```

Typical block settings:

- LLaMA-2 7B, LLaMA-3 8B, LLaMA-2 13B: `block_diag=2`
- LLaMA-2 70B, LLaMA-3 70B: `block_diag=4`

## Run LAC Evaluation

```bash
sh scripts/eval_lac.sh <MODEL_PATH> <ROTATION_PATH> <USE_BF16> <CALI_BS> <W_BITS> <A_BITS> <KV_BITS>
```

For 70B models, reduce calibration batch size if GPU memory is tight. The original experiments used about 24 GB GPU memory, but CUDA/runtime overhead can vary.

## Results

Perplexity on WikiText2 and average accuracy across nine zero-shot tasks. All models are evaluated using uniform quantization with InfoQuant.

| #Bits (W-A-KV) | Method    | LLaMA-3 8B | Wiki | LLaMA-2 7B | Wiki | LLaMA-2 13B | Wiki | LLaMA-2 70B | Wiki | LLaMA-3 70B | Wiki |
| -------------- | --------- | ---------- | ---- | ---------- | ---- | ----------- | ---- | ----------- | ---- | ----------- | ---- |
| 16-16-16       | FP16      | 68.09      | 6.14 | 65.21      | 5.47 | 67.61       | 4.88 | 71.59       | 3.32 | 73.81       | 2.86 |
| 4-16-16        | InfoQuant | 67.36      | 6.48 | 64.34      | 5.60 | 67.27       | 4.99 | 71.25       | 3.40 | 73.25       | 3.50 |
| 4-4-16         | InfoQuant | 65.74      | 7.07 | 62.84      | 5.86 | 66.71       | 5.15 | 70.82       | 3.62 | 70.71       | 5.24 |
| 4-4-4          | InfoQuant | 65.57      | 7.16 | 63.16      | 5.89 | 66.33       | 5.18 | 70.35       | 3.64 | 70.21       | 5.39 |

Baseline results for RTN, SmoothQuant, GPTQ, AWQ, QuaRot, SpinQuant, and OSTQuant are referenced from:

> OSTQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting
> Xing Hu, Yuan Cheng, Dawei Yang, Zhixuan Chen, Zukang Xu, Jiangyong Yu, XUCHEN, Zhihang Yuan, Zhe Jiang, Sifan Zhou
> The Thirteenth International Conference on Learning Representations (ICLR), 2025
> https://openreview.net/forum?id=rAcgDBdKnP
