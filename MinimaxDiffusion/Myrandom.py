import os
import shutil
import random

def copy_random_files(source_path, destination_path, num_files=50):
    # 获取源文件夹中的所有文件
    all_files = os.listdir(source_path)

    # 从文件列表中随机选择指定数量的文件
    selected_files = random.sample(all_files, min(num_files, len(all_files)))

    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # 复制选定的文件到目标文件夹
    for file_name in selected_files:
        source_file = os.path.join(source_path, file_name)
        destination_file = os.path.join(destination_path, file_name)
        shutil.copyfile(source_file, destination_file)

folder_list=["n01440764", "n02102040", "n02979186", "n03000684", "n03028079", "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]
# 指定源文件夹和目标文件夹的路径

for folder in folder_list:
    source_folder_path = os.path.join("/root/autodl-tmp/MinimaxDiffusion/data/imagenet/train/",folder)
    destination_folder_path = os.path.join("/root/autodl-tmp/results/nette_random",folder)

    # 调用函数复制随机文件
    copy_random_files(source_folder_path, destination_folder_path, num_files=50)
