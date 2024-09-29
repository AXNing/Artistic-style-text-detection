import os
import cv2
import numpy as np

def process_and_save_rectangles(input_directory, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历输入目录下的所有txt文件
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")

            # 处理当前txt文件
            process_txt_file(input_filepath, output_filepath)

def process_txt_file(input_filepath, output_filepath):
    # 读取txt文件
    with open(input_filepath, 'r') as file:
        lines = file.readlines()

    # 处理每一行，获取标注框的坐标信息（忽略最后一列）
    with open(output_filepath, 'w') as output_file:
        for line in lines:
            # 将每行数据分割成坐标和标签
            parts = line.strip().split(',')

            # 获取坐标部分（忽略最后一列）
            coordinates = [float(coord) for coord in parts[:-1]]

            # 将坐标信息转换为NumPy数组
            points = np.array(coordinates).reshape(-1, 2)

            # 计算最小外接矩形
            rect = cv2.minAreaRect(np.int32(points))
            # 获取矩形的四个角坐标
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cx, cy = rect[0]
            w_, h_ = rect[1]
            angle = rect[2]
            mid_ = 0
            if angle > 45:
                angle = 90 - angle
                mid_ = w_
                w_ = h_
                h_ = mid_
            elif angle < -45:
                angle = 90 + angle
                mid_ = w_
                w_ = h_
                h_ = mid_
            angle = angle / 180 * 3.141592653589

            x_min = int(cx - w_ / 2)
            x_max = int(cx + w_ / 2)
            y_min = int(cy - h_ / 2)
            y_max = int(cy + h_ / 2)

            # 写入新的txt文件
            output_file.write(f"{x_min},{y_min},{x_max},{y_max},{angle}\n")

# 使用示例
input_directory = "/home/ubuntu/axproject/TextBPN-Plus-Plus-main_1/data/haibao/Test/gt"
output_directory = "/home/ubuntu/axproject/TextBPN-Plus-Plus-main_1/all_eval"

process_and_save_rectangles(input_directory, output_directory)