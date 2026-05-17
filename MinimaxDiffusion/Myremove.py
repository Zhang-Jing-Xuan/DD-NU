import os
# path="/root/autodl-tmp/results/dit-distillation"
# for i in range(39,40):
#     path_distill=os.path.join(path,f"1k_10_{i}-distill")
#     path_mini = os.path.join(path,f"1k_10_{i}-minimax")
#     os.system(f"rm -rf {path_distill}")
#     os.system(f"rm -rf {path_mini}")

# path="/root/autodl-tmp/MinimaxDiffusion/pretrained_models"
# for i in range(0,40):
#     path_model=os.path.join(path,f"latent_10_{i}")
#     os.system(f"rm -rf {path_model}")

path = "/root/autodl-tmp/MinimaxDiffusion/results/latent/"
for i in range(0,40):
    path_distill=os.path.join(path,f"conv3in_grad10_750_4_1k_10_{i}_grad_mse_pt2_nd2000_inloop10_cut_niter100_factor1_lr0.05_random_ipc10_phase{i}")
    os.system(f"rm -rf {path_distill}")