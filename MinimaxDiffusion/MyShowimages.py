import os
from PIL import Image

# 定义图片文件夹路径
path = "/root/autodl-tmp/results/dit-distillation/1k-distill"
output_image_path = '/root/autodl-tmp/output_image_with_padding_1.pdf'
image_folder = os.listdir(path)
image_list=[]
for i in range(250):
    folder_i = image_folder[i]
    image = os.listdir(os.path.join(path,folder_i))[0]
    image_list.append(os.path.join(path,folder_i,image))
# image_list = image_list[-75:-65]
# 获取图片文件列表
# image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 设置网格大小
grid_rows = 10
grid_cols = 25

# 设置每张图片的尺寸 (假设每张图片大小相同)
thumb_width, thumb_height = 64, 64  # 每张缩略图的大小
padding = 1  # 图片之间的间距

# 创建一个新图像，用于存放拼接结果，加上间距
output_image = Image.new('RGB', 
                         (grid_cols * (thumb_width + padding) - padding, 
                          grid_rows * (thumb_height + padding) - padding), 
                         color=(0, 0, 0))  # 背景设为黑色

# 逐个读取图片并粘贴到网格中
for i, image_file in enumerate(image_list):
    img_path = image_file
    img = Image.open(img_path)
    img = img.resize((thumb_width, thumb_height))  # 调整图片大小

    # 计算图片在网格中的位置，考虑间距
    x = (i % grid_cols) * (thumb_width + padding)
    y = (i // grid_cols) * (thumb_height + padding)

    # 粘贴图片到输出图像
    output_image.paste(img, (x, y))

# 保存结果到本地
output_image.save(output_image_path)
print(f"Image with padding saved at {output_image_path}")
