import os
import shutil

# 定义文件夹路径
folder_1 = '/root/autodl-tmp/results/dit-distillation/1k_500_1-10-minimax-distill/'
folder_2 = '/root/autodl-tmp/results/dit-distillation/1k_500_2-10-minimax-distill/'
folder = "/root/autodl-tmp/results/dit-distillation/1k-distill"
image_folder = os.listdir(folder)
print(len(image_folder))
      
# # 定义一个函数将指定源文件夹中的所有子文件夹移动到目标文件夹
# def move_subfolders_to_target(src_folder, target_folder):
#     for item in os.listdir(src_folder):
#         src_item_path = os.path.join(src_folder, item)
#         # target_item_path = os.path.join(target_folder, item)
#         target_item_path = target_folder
#         if os.path.isdir(src_item_path):
#             shutil.move(src_item_path, target_item_path)
#             print(f"Moved {src_item_path} to {target_item_path}")

# move_subfolders_to_target(folder_1, folder)
# move_subfolders_to_target(folder_2, folder)

# print("All subfolders moved to folder.")