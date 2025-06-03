# 🐾 AniMo: Species-Aware Model for Text-Driven Animal Motion Generation (CVPR 2025) 🦁

<!-- [![arXiv](https://img.shields.io/badge/arXiv-xxxx.xx-b31b1b.svg)](https://arxiv.org/abs/xxx.xxx) -->

[![cvpr2025](https://img.shields.io/badge/🏆-CVPR%202025%20Main%20Conference-1b427d)](https://cvpr.thecvf.com/virtual/2025/poster/34318)
[![cv4animal](https://img.shields.io/badge/🦁-CV4Animals%20Workshop%202025-brightgreen)](https://www.cv4animals.com/)
![Stars](https://img.shields.io/github/stars/WandererXX/AniMo)

# 🚀 Getting Started

## 🛠️ Environment

```bash
conda env create -f environment.yml
conda activate animo
pip install git+https://github.com/openai/CLIP.git
```

### 📊 Data

Please follow the instructions in `./data_generation/README.md` to obtain the AniMo4D dataset. 📂

# 🏋️‍♂️ Training AniMo Model

You may also need to download evaluation models in `./text_mot_match/README.md` and glove files in `./glove/README.md` to run the scripts. ⚙️

## 🔢 Train RVQ

```bash
python train_vq.py \
  --name rvq \
  --gpu_id 0 \
  --batch_size 128 \
  --num_quantizers 6 \
  --max_epoch 50 \
  --quantize_dropout_prob 0.2 \
  --gamma 0.05 \
  --warm_up_iter 80000 \
  --checkpoints_dir ckpt/animo
```

## 🎭 Train Masked Transformer

```bash
python train_t2m_transformer.py \
  --name mtrans \
  --gpu_id 0 \
  --batch_size 64 \
  --vq_name rvq \
  --max_epoch 80 \
  --checkpoints_dir ckpt/animo
```

## ➕ Train Residual Transformer

```bash
python train_res_transformer.py \
  --name rtrans \
  --gpu_id 0 \
  --batch_size 64 \
  --vq_name rvq \
  --cond_drop_prob 0.2 \
  --max_epoch 80 \
  --share_weight \
  --checkpoints_dir ckpt/animo
```

# 📈 Evaluation

## 🔍 Evaluate RVQ Reconstruction:

```bash
python eval_t2m_vq.py \
  --gpu_id 0 \
  --name rvq \
  --ext eval_reconstruction \
  --checkpoints_dir ckpt/animo
```

## ✨ Evaluate Text2motion Generation:

```bash
python eval_t2m_trans_res.py \
  --res_name rtrans \
  --name mtrans \
  --vq_name rvq \
  --gpu_id 0 \
  --cond_scale 4 \
  --time_steps 10 \
  --ext eval_generation \
  --which_epoch latest \
  --checkpoints_dir ckpt/animo
```

# 💖 Acknowlegements

We sincerely thank the open-sourcing of these works where our code is based on:
[MoMask](https://github.com/EricGuo5513/momask-codes), [Text-to-motion](https://github.com/EricGuo5513/text-to-motion), [AttT2M](https://github.com/ZcyMonkey/AttT2M), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), and [T2M-GPT](https://github.com/Mael-zys/T2M-GPT).
