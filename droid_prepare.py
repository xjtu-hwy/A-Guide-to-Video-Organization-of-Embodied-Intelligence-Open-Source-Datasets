import tensorflow_datasets as tfds
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm

def count_files_in_folder(folder_path):
    file_count = 0
    try:
        for root, dirs, files in os.walk(folder_path):
            file_count += len(files)
        return file_count
    except FileNotFoundError:
        print(f"错误: 未找到指定的文件夹 {folder_path}。")
        return None
    except PermissionError:
        print(f"错误: 没有权限访问文件夹 {folder_path}。")
        return None





def save_images_as_video(images, video_path, fps=30, width=704, height_ratio=180/320, min_frames=34):
    """ 将一系列图像保存为视频文件，并调整视频分辨率，确保视频至少包含 min_frames 帧 """
    height = int(width * height_ratio)  # 根据给定比例调整高度
    frame_size = (width, height)  # 设置帧的尺寸

    # 如果帧数少于 min_frames，补充静止帧
    if len(images) < min_frames:
        first_frame = images[0]
        while len(images) < min_frames:
            images.insert(0, first_frame)  # 在开头添加静止帧

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for img in images:
        # 将图像调整为设定的分辨率
        img_resized = img.resize((width, height))

        # 将图像转换为BGR格式，以适应OpenCV
        img_bgr = np.array(img_resized)[:, :, ::-1]  # RGB -> BGR
        video_writer.write(img_bgr)

    video_writer.release()


def save_language_instructions_and_videos(dataset, s, save_dir="/home/datasets", prefix="file_", checkpoint_file="checkpoint.txt"):
    """ 提取所有 language_instruction, language_instruction_2, language_instruction_3 并保存为 .txt 文件，同时保存为 .mp4 文件 """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(s):
        os.makedirs(s)

    process_file = os.path.join(s, "droid_process.txt")

    # 恢复进度
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            file_counter = int(f.read().strip())  # 读取上次保存的最大 file_counter
    else:
        file_counter = 0  # 如果没有进度文件，从0开始
    file_counter = 0
    check_filecounter = 0
    # 遍历整个数据集
    for idx, episode in enumerate(tqdm(dataset, desc="Processing episodes", initial=file_counter, total=len(dataset))):
        # 写入当前 idx 到 droid_process.txt
        with open(process_file, "w") as f:
            f.write(str(idx))

        if idx < file_counter:  # 如果已经处理过这个 episode，跳过
            continue

        exterior_image_1_left_images = []
        exterior_image_2_left_images = []
        wrist_image_left_images = []

        # 合并所有 language_instruction 字段
        full_language_instruction = ""  # 用来保存合并后的语言指令
        language_instruction = None

        for step in episode["steps"]:
            # 判断是否存在有效的 language_instruction
            if step['language_instruction'] != b'':
                language_instruction = step['language_instruction']
            elif step['language_instruction_2'] != b'':
                language_instruction = step['language_instruction_2']
            elif step['language_instruction_3'] != b'':
                language_instruction = step['language_instruction_3']

            if language_instruction:
                # 合并有效的 language_instruction 字段
                full_language_instruction = language_instruction
                break  # 只取第一个有效的 language_instruction

        # 如果没有有效的 language_instruction，则跳过
        if not language_instruction:
            continue
        else:
            check_filecounter += 3
            if check_filecounter < total_files:
                file_counter = check_filecounter
                continue

        # 提取图像并保存
        for step in episode["steps"]:
            exterior_image_1_left = Image.fromarray(step["observation"]["exterior_image_1_left"].numpy())
            exterior_image_2_left = Image.fromarray(step["observation"]["exterior_image_2_left"].numpy())
            wrist_image_left = Image.fromarray(step["observation"]["wrist_image_left"].numpy())

            exterior_image_1_left_images.append(exterior_image_1_left)
            exterior_image_2_left_images.append(exterior_image_2_left)
            wrist_image_left_images.append(wrist_image_left)

        # 保存为视频（视频文件名现在只基于 file_counter）
        txt_path = os.path.join(save_dir, f"{prefix}{file_counter}.txt")
        with open(txt_path, "w") as f:
            f.write(f"{full_language_instruction}")
        with open(txt_path, "r+") as f:
            content = f.read()
            f.seek(0)
            f.write(content[2:-1])
            f.truncate()
        video_filename_1 = f"{prefix}{file_counter}.mp4"
        file_counter += 1

        txt_path = os.path.join(save_dir, f"{prefix}{file_counter}.txt")
        with open(txt_path, "w") as f:
            f.write(f"{full_language_instruction}")
        with open(txt_path, "r+") as f:
            content = f.read()
            f.seek(0)
            f.write(content[2:-1])
            f.truncate()
        video_filename_2 = f"{prefix}{file_counter}.mp4"
        file_counter += 1

        txt_path = os.path.join(save_dir, f"{prefix}{file_counter}.txt")
        with open(txt_path, "w") as f:
            f.write(f"{full_language_instruction}")
        with open(txt_path, "r+") as f:
            content = f.read()
            f.seek(0)
            f.write(content[2:-1])
            f.truncate()
        video_filename_3 = f"{prefix}{file_counter}.mp4"
        file_counter += 1

        video_path_1 = os.path.join(save_dir, video_filename_1)
        video_path_2 = os.path.join(save_dir, video_filename_2)
        video_path_3 = os.path.join(save_dir, video_filename_3)

        save_images_as_video(exterior_image_1_left_images, video_path_1)
        save_images_as_video(exterior_image_2_left_images, video_path_2)
        save_images_as_video(wrist_image_left_images, video_path_3)

        # 保存当前进度
        with open(checkpoint_file, "w") as f:
            f.write(str(file_counter + 1))

        # 更新文件计数器


# 用户可以在此处设置保存的文件夹路径
save_directory = "/home/datasets/droid"  # 你可以修改这个路径
prefix = "droid_viedo"  # 文件名前缀
s = "/home/hongweiyi/dataset_process" # 独立文件夹地址
your_droid_path = '/home/datasets/droid/1.0.0'


# 加载数据集并处理
ds = tfds.builder_from_directory(your_droid_path).as_dataset(split='all')
save_language_instructions_and_videos(ds, s, save_dir=save_directory, prefix=prefix)
    