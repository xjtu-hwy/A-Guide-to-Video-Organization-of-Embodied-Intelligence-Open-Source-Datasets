import h5py
import cv2
import numpy as np
import os
import re  # 导入正则表达式模块，用于处理txt内容


def extract_caption_from_filename(filename):
    """
    从文件名中提取caption（文件名去除扩展名和下划线）
    
    :param filename: .hdf5文件路径
    :return: 文件的caption（文件名去除扩展名和下划线，并删除最后5个字符）
    """
    name = os.path.basename(filename)
    caption = name.replace(".hdf5", "").replace("_", " ")
    # 删除最后5个字符
    caption = caption[:-5] if len(caption) >= 5 else caption  # 确保字符串长度至少为5
    return caption


def process_and_save_video(demo, save_dir, file_counter, prefix):
    """
    处理并保存视频（对视频进行补帧处理）
    
    :param demo: 单个demo的hdf5数据
    :param save_dir: 保存视频的文件夹路径
    :param file_counter: 用于命名文件的计数器
    :param prefix: 文件命名的前缀
    """
    # 提取数据
    agentview_rgb = demo['obs']['agentview_rgb'][:]
    eye_in_hand_rgb = demo['obs']['eye_in_hand_rgb'][:]
    
    # 检查视频每部分的总帧数
    # 补帧处理：如果低于34帧就在视频之前补静止帧
    if len(agentview_rgb) < 34:
        # 计算需要补的帧数
        frames_to_add = 34 - len(agentview_rgb)
        # 复制视频的第一帧作为补帧
        first_frame = agentview_rgb[0:1]  # 获取第一帧
        agentview_rgb = np.repeat(first_frame, frames_to_add, axis=0)  # 复制第一帧多次
        agentview_rgb = np.concatenate((agentview_rgb, demo['obs']['agentview_rgb'][:]), axis=0)  # 拼接原视频
    
    if len(eye_in_hand_rgb) < 34:
        # 计算需要补的帧数
        frames_to_add = 34 - len(eye_in_hand_rgb)
        # 复制视频的第一帧作为补帧
        first_frame = eye_in_hand_rgb[0:1]  # 获取第一帧
        eye_in_hand_rgb = np.repeat(first_frame, frames_to_add, axis=0)  # 复制第一帧多次
        eye_in_hand_rgb = np.concatenate((eye_in_hand_rgb, demo['obs']['eye_in_hand_rgb'][:]), axis=0)  # 拼接原视频
    
    # 获取视频帧的尺寸
    frame_height, frame_width = agentview_rgb.shape[1], agentview_rgb.shape[2]
    
    # 设置目标分辨率
    target_height, target_width = 704, 704
    
    # 设置视频编码器（以30帧/秒保存视频）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 创建视频写入对象
    video_path_before = os.path.join(save_dir, f"{prefix}_viedo{file_counter}.mp4")
    video_path_after = os.path.join(save_dir, f"{prefix}_viedo{file_counter+1}.mp4")
    
    out_before = cv2.VideoWriter(video_path_before, fourcc, 30.0, (target_width, target_height))
    out_after = cv2.VideoWriter(video_path_after, fourcc, 30.0, (target_width, target_height))
    
    # 将每一帧写入视频
    for frame in agentview_rgb:
        # 缩放到目标尺寸
        frame_resized = cv2.resize(frame, (target_width, target_height))
        out_before.write(frame_resized)
    out_before.release()

    for frame in eye_in_hand_rgb:
        # 缩放到目标尺寸
        frame_resized = cv2.resize(frame, (target_width, target_height))
        out_after.write(frame_resized)
    out_after.release()


def save_caption_to_txt(caption, save_dir, file_counter, prefix):
    """
    保存demo的caption到txt文件中
    
    :param caption: 从文件名提取的caption
    :param save_dir: 保存txt文件的文件夹路径
    :param file_counter: 用于命名文件的计数器
    :param prefix: 文件命名的前缀
    """
    caption_file_path = os.path.join(save_dir, f"{prefix}_viedo{file_counter}.txt")
    # 提取核心内容（删除前面的"Caption: "和后面的" demo"部分）
    core_content = re.sub(r'^Caption:\s*(.*?)\s*demo$', r'\1', caption)
    with open(caption_file_path, 'w') as txt_file:
        txt_file.write(f"{core_content}\n")


def process_hdf5_to_video_and_txt(file_path, save_dir, prefix, file_counter):
    """
    处理.hdf5文件，将每个demo的数据保存为视频和caption文本
    
    :param file_path: .hdf5文件路径
    :param save_dir: 保存视频和txt文件的文件夹路径
    :param prefix: 文件命名的前缀
    :param file_counter: 用于命名文件的计数器
    :return: 更新后的file_counter
    """
    # 确保保存的文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 打开HDF5文件
    with h5py.File(file_path, 'r') as f:
        # 进入data组
        data_group = f['data']
        
        # 提取caption信息（从文件名提取）
        caption = extract_caption_from_filename(file_path)

        # 遍历每个demo
        for demo_key in data_group.keys():
            demo = data_group[demo_key]
            
            # 检查是否有 'obs' 组
            if 'obs' not in demo:
                print(f"Warning: 'obs' group does not exist for demo {demo_key}. Skipping...")
                continue

            # 保存视频
            process_and_save_video(demo, save_dir, file_counter, prefix)

            # 保存caption到txt文件
            save_caption_to_txt(caption, save_dir, file_counter, prefix)
            save_caption_to_txt(caption, save_dir, file_counter+1, prefix)

            print(f"Demo {demo_key} videos and caption saved in {save_dir}.")
            
            # 更新file_counter
            file_counter += 2

    return file_counter  # 返回更新后的file_counter


def process_folder(folder_path, save_dir, prefix, file_counter=0):
    """
    处理文件夹中的所有.hdf5文件
    
    :param folder_path: 包含.hdf5文件的文件夹路径
    :param save_dir: 保存视频和txt文件的文件夹路径
    :param prefix: 文件命名的前缀
    :param file_counter: 用于命名文件的计数器
    :return: 更新后的file_counter
    """
    # 获取文件夹中的所有.hdf5文件
    hdf5_files = [f for f in os.listdir(folder_path) if f.endswith('.hdf5') and not f.startswith('.')]

    # 遍历文件夹中的每个.hdf5文件
    for hdf5_file in sorted(hdf5_files):
        file_path = os.path.join(folder_path, hdf5_file)
        print(f"Processing file: {file_path}")
        file_counter = process_hdf5_to_video_and_txt(file_path, save_dir, prefix, file_counter)

    return file_counter  # 返回更新后的file_counter


# 调用函数处理文件夹中的所有.hdf5文件
folder_path_list = [
    '/home/datasets/libero/libero_100/libero_10',  # 第一个文件夹路径
    '/home/datasets/libero/libero_100/libero_90'
]  # 替换为包含.hdf5文件的文件夹路径列表

save_folder_list = [
    '/home/datasets/viedo2qzz/libero/100/10',  # 第一个保存文件夹路径
    '/home/datasets/viedo2qzz/libero/100/90'
]  # 替换为保存文件的文件夹路径列表，与folder_path_list一一对应

prefix_list = ['libero_10',
          'libero_90']  # 自定义文件命名的前缀

for folder_path, save_dir, prefix in zip(folder_path_list, save_folder_list, prefix_list):
    print(f"Processing folder '{folder_path}' and saving to '{save_dir}'")
    file_counter = 0  # 每个文件夹的处理都从计数器0开始
    file_counter = process_folder(folder_path, save_dir, prefix, file_counter)