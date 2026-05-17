# Dataset Distillation via a Noise-Unconstrained Generative Model

## Highlights :sparkles:
- During the distillation process, an adaptive coefficient is introduced for small-resolution datasets and the improved MiniMax loss is adopted for large-resolution datasets.
- During the deployment process, online distillation is adopted for small-resolution datasets and offline distillation is adopted for large-resolution datasets.

![Basic idea](assets/arch.png)

## Getting Started
### Installation
```bash
conda create -n diff python=3.8
conda activate diff
pip install -r requirements.txt
```

## Experiment Commands
### For small-resolution datasets (SVHN, CIFAR10), run the following command:
1. Enter into the DiM directory.
```bash
cd DiM
```

2. Train a conditional GAN model.
```bash
CUDA_VISIBLE_DEVICES=0 python gen_condense.py --tag gan
```

3. Select the best model for further matching.
```bash
CUDA_VISIBLE_DEVICES=0 python pool_match.py --tag match --match logit --match-aug --weight [BEST_MODEL]
```

4. Validate the generator performance.
```bash
CUDA_VISIBLE_DEVICES=0 python validate_all.py --tag test --pretrain-weight [BEST_MODEL]
```

### For the subsets of ImageNet1K datasets, run the following command:
1. Enter into the MinimaxDiffusion directory.
```bash
cd MinimaxDiffusion
```
2. Run train_dit.py to fine-tune the diffusion model. Models and log files are stored in /root/autodl-tmp/logs.
```bash
torchrun --nnode=1 --master_port=25678 train_dit_myrd.py --model DiT-XL/2 \
     --data-path /root/autodl-tmp/MinimaxDiffusion/data/imagenet/train/ --ckpt pretrained_models/DiT-XL-2-256x256.pt \
     --global-batch-size 8 --tag minimax --ckpt-every 12000 --log-every 1500 --epochs 8 \
     --condense --finetune-ipc -1 --results-dir ../logs/run-0-myrd_woof --spec woof
```
3. Run sample_latent.py to sample latent vectors. Results are stored in /root/autodl-tmp/results/dit-distillation. "--num-samples" specifies the number of samples.
```bash
python sample_latent.py --model DiT-XL/2 --image-size 256 --ckpt ../logs/run-0-myrd_woof/000-DiT-XL-2-minimax/checkpoints/0012000.pt --save-dir ../results/dit-distillation/imagewoof-myrd-1000 --spec woof --num-samples 1000
```

4. Run pretrain_latent.py to obtain pre-trained weights. You need to update the latent vector sampling path obtained in Step 2.
```bash
python pretrain_latent.py -d latent_woof --nclass 10 -n convnet --depth 3  --pt_from 2 --seed 2023
```

5. Update the log file path for self.logger in misc/utils.py.

6. Run condense_latent.py to condense latent vectors and evaluate the results.
```bash
python condense_latent.py --reproduce -d latent_woof -f 1 --ipc 10 -n convnet --depth 3 --model_path "/root/autodl-tmp/MinimaxDiffusion/pretrained_models/latent_woof/conv3in_cut_seed_2023_lr_0.01_aug_color_crop_cutout/" --niter 300 --inner_loop 10 --tag grad10_woof_myrd
```
- Results are stored in /root/autodl-tmp/results/dit-distillation with a distill suffix.
- condense_latent.py automatically decodes the condensed latent vectors into images and performs cross-architecture evaluation.
- You need to update the paths for: pre-trained weights, latent vector samples, condensed latent vectors, and the fine-tuned diffusion model weights.


## Acknowledgement
This project is mainly developed based on the following works:
- [DiM](https://github.com/vimar-gu/DiM)
- [MiniMax](https://github.com/vimar-gu/MinimaxDiffusion)
- [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)

## Citation
If you find this work helpful, please cite:
```
@ARTICLE{11509640,
  author={Zhang, Jingxuan and Dai, Lei and Ye, Fei and Chen, Zhihua and Li, Ping and Yang, Xiaokang and Sheng, Bin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Dataset Distillation via a Noise-Unconstrained Generative Model}, 
  year={2026},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2026.3690778}
}
```
