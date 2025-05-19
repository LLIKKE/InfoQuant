# InfoQuant

Official implementation for the paper **InfoQuant: Reducing Quantization Information Error via Peak Suppression Orthogonal Transformation**.

---

## 🚀 Getting Started

### 1. Environment Setup

- Python ≥ 3.9  
- PyTorch ≥ 2.0

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Install the [Fast Hadamard Transform](https://github.com/Dao-AILab/fast-hadamard-transform):

```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git  
cd fast-hadamard-transform  
pip install .
```

---

### 2. Download Model Weights

We recommend downloading LLaMA-2 or LLaMA-3 model weights locally from [Hugging Face](https://huggingface.co/meta-llama):

---

### 3. Run PSOT: Optimize the Orthogonal Rotation Matrix

Use the script `get_matrix.sh` to optimize the orthogonal rotation matrix. Replace `<MODEL_PATH>` with the path to your model weights.

```bash
sh scripts/get_matrix.sh <MODEL_PATH> <BLOCK_NUM> <W_BITS> <A_BITS> <KV_BITS>
```

#### Examples:

- **LLaMA-2 7B / LLaMA-3 8B / LLaMA-2 13B**
  ```bash
  sh scripts/get_matrix.sh <MODEL_PATH> 2 4 4 4
  ```


- **LLaMA-2 70B / LLaMA-3 70B**
  ```bash
  sh scripts/get_matrix.sh <MODEL_PATH> 4 4 4 4
  ```

---

### 4. Evaluate with Learning Activation Clipping (LAC)

Once the rotation matrix is generated, place it in the appropriate `ROTATION_PATH` and run the evaluation script:

```bash
sh scripts/eval_lac.sh <MODEL_PATH> <ROTATION_PATH> <USE_BF16> <CAIL_BS> <W_BITS> <A_BITS> <KV_BITS>
```

#### Examples:

- **LLaMA-2 7B / LLaMA-3 8B / LLaMA-2 13B**
  ```bash
  sh scripts/eval_lac.sh <MODEL_PATH> <ROTATION_PATH> False 4 4 4 4
  ```

- **LLaMA-2 70B**
  ```bash
  sh scripts/eval_lac.sh <MODEL_PATH> <ROTATION_PATH> False 3 4 4 4
  ```

- **LLaMA-3 70B**
  ```bash
  sh scripts/eval_lac.sh <MODEL_PATH> <ROTATION_PATH> True 3 4 4 4
  ```
  
> **Note:** Quantizing 70B models typically requires around **24 GB** of GPU memory. However, due to potential fluctuations in memory usage (e.g., from CUDA kernel launches or framework overhead), we recommend **reducing the batch size by 1** to ensure stable execution **within a strict 24 GB memory limit**.

---

## 📊 Main Results

Perplexity on WikiText2 and average accuracy across nine zero-shot tasks. All models are evaluated using uniform quantization with our InfoQuant method.

| #Bits (W-A-KV) | Method    | LLaMA-3 8B |       | LLaMA-2 7B |       | LLaMA-2 13B |       | LLaMA-2 70B |       | LLaMA-3 70B |       |
|----------------|-----------|------------|-------|------------|-------|-------------|-------|--------------|-------|--------------|-------|
|                |           | 0-shot ↑   | Wiki ↓ | 0-shot ↑   | Wiki ↓ | 0-shot ↑    | Wiki ↓ | 0-shot ↑     | Wiki ↓ | 0-shot ↑     | Wiki ↓ |
| **16-16-16**    | FP16      | 68.09               | 6.14   | 65.21               | 5.47   | 67.61                 | 4.88   | 71.59                 | 3.32   | 73.81                 | 2.86   |
| **4-16-16**     | InfoQuant | 67.36               | 6.48   | 64.34               | 5.60   | 67.27                 | 4.99   | 71.25                 | 3.40   | 73.25                 | 3.50   |
| **4-4-16**      | InfoQuant      | 65.74               | 7.07   | 62.84               | 5.86   | 66.71                 | 5.15   | 70.82                 | 3.62   | 70.71                 | 5.24   |
| **4-4-4**       | InfoQuant      | 65.57               | 7.16   | 63.16               | 5.89   | 66.33                 | 5.18   | 70.35                 | 3.64   | 70.21                 | 5.39   |

The baseline results, including **RTN**, **SmoothQuant**, **GPTQ**, **AWQ**, **QuaRot**, **SpinQuant**, and **OSTQuant**, are referenced from the paper:

> **OSTQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting**  
> Xing Hu, Yuan Cheng, Dawei Yang, Zhixuan Chen, Zukang Xu, Jiangyong Yu, XUCHEN, Zhihang Yuan, Zhe Jiang, Sifan Zhou  
> *The Thirteenth International Conference on Learning Representations (ICLR), 2025*  
> [https://openreview.net/forum?id=rAcgDBdKnP](https://openreview.net/forum?id=rAcgDBdKnP)

Please refer to this work for a comprehensive comparison and implementation details of these baselines.

