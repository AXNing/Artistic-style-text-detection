import os
import cv2
import numpy as np
import subprocess
from cfglib.config import config as cfg
from util.misc import mkdirs


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def analysize_result(source_dir, fid_path, outpt_dir, name):

    bad_txt = open("{}/eval.txt".format(outpt_dir), 'w')#打开一个名为 eval.txt 的文件，用于记录一些可能的不良结果
    all_eval = open("{}/{}/{}_eval.txt".format(cfg.output_dir, "Analysis", name), 'a+')#打开一个名为 {name}_eval.txt 的文件，用于记录整体的评估结果。文件模式为 'a+' 表示以追加方式打开，如果文件不存在则创建。
    sel_list = list()
    with open(fid_path) as f:
        lines = f.read().split("\n")
        for line in lines:
            line_items = line.split(" ")
            id = line_items[0]
            precision = float(line_items[2].split('=')[-1])
            recall = float(line_items[4].split('=')[-1])
            if id != "ALL" and (precision < 0.5 or recall < 0.5):
                img_path = os.path.join(source_dir, line_items[0].replace(".txt", ".jpg"))
                if os.path.exists(img_path):
                    os.system('cp {} {}'.format(img_path, outpt_dir))
                sel_list.append((int(id.replace(".txt", "").replace("img", "").replace("_", "")), line))
            if id == "ALL":
                all_eval.write("{} {} {}\n".format(
                    outpt_dir.split('/')[-1],
                    "{}/{}".format(cfg.dis_threshold, cfg.cls_threshold),
                    line))
    sel_list = sorted(sel_list, key=lambda its: its[0])
    bad_txt.write('\n'.join([its[1] for its in sel_list]))
    all_eval.close()
    bad_txt.close()


def deal_eval_total_text(debug=False):
    # compute DetEval
    eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(
        ['python', 'dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py', cfg.exp_name, '--tr', '0.7',
         '--tp', '0.6'])
    subprocess.call(
        ['python', 'dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py', cfg.exp_name, '--tr', '0.8',
         '--tp', '0.4'])

    if debug:
        source_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
        outpt_dir_base = os.path.join(cfg.output_dir, "Analysis", "eval_view", "total_text")
        if not os.path.exists(outpt_dir_base):
            mkdirs(outpt_dir_base)

        outpt_dir1 = os.path.join(outpt_dir_base, "{}_{}_{}_{}_{}"
                                  .format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch, 0.7, 0.6))
        osmkdir(outpt_dir1)
        fid_path1 = '{}/Eval_TotalText_{}_{}.txt'.format(eval_dir, 0.7, 0.6)

        analysize_result(source_dir, fid_path1, outpt_dir1, "totalText")

        outpt_dir2 = os.path.join(outpt_dir_base, "{}_{}_{}_{}_{}"
                                  .format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch, 0.8, 0.4))
        osmkdir(outpt_dir2)
        fid_path2 = '{}/Eval_TotalText_{}_{}.txt'.format(eval_dir, 0.8, 0.4)

        analysize_result(source_dir, fid_path2, outpt_dir2, "totalText")

    print('End.')


def deal_eval_ctw1500(debug=False):
    # compute DetEval
    eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(['python', 'dataset/ctw1500/Evaluation_Protocol/ctw1500_eval.py', cfg.exp_name])

    if debug:
        source_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
        outpt_dir_base = os.path.join(cfg.output_dir, "Analysis", "eval_view", "ctw1500")
        if not os.path.exists(outpt_dir_base):
            mkdirs(outpt_dir_base)

        outpt_dir = os.path.join(outpt_dir_base, "{}_{}_{}".format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch))
        osmkdir(outpt_dir)
        fid_path1 = '{}/Eval_ctw1500_{}.txt'.format(eval_dir, 0.5)

        analysize_result(source_dir, fid_path1, outpt_dir, "ctw1500")

    print('End.')


def deal_eval_icdar15(debug=False):
    # compute DetEval
    eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    input_dir = 'output/{}'.format(cfg.exp_name)
    father_path = os.path.abspath(input_dir)
    print(father_path)
    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(['sh', 'dataset/icdar15/eval.sh', father_path])

    if debug:
        source_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
        outpt_dir_base = os.path.join(cfg.output_dir, "Analysis", "eval_view", "icdar15")
        if not os.path.exists(outpt_dir_base):
            mkdirs(outpt_dir_base)

        outpt_dir = os.path.join(outpt_dir_base, "{}_{}_{}".format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch))
        osmkdir(outpt_dir)
        fid_path1 = '{}/Eval_icdar15.txt'.format(eval_dir)

        analysize_result(source_dir, fid_path1, outpt_dir, "icdar15")

    print('End.')

    pass


def deal_eval_TD500(debug=False):
    # compute DetEval
    eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    input_dir = 'output/{}'.format(cfg.exp_name)
    father_path = os.path.abspath(input_dir)
    print(father_path)
    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(['sh', 'dataset/TD500/eval.sh', father_path])#subprocess.call 调用了一个 shell 脚本 (eval.sh)，该脚本位于 'dataset/TD500/' 目录下，同时传递了 father_path 作为参数。这个脚本可能用于执行性能评估的一些操作

    if debug:
        source_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
        outpt_dir_base = os.path.join(cfg.output_dir, "Analysis", "eval_view", "TD500")
        if not os.path.exists(outpt_dir_base):
            mkdirs(outpt_dir_base)

        outpt_dir = os.path.join(outpt_dir_base, "{}_{}_{}".format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch))
        osmkdir(outpt_dir)
        fid_path1 = '{}/Eval_TD500.txt'.format(eval_dir)

        analysize_result(source_dir, fid_path1, outpt_dir, "TD500")

    print('End.')
def deal_eval_Movie_Poster(debug=False):
    # compute DetEval
    eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    input_dir = 'output/{}'.format(cfg.exp_name)
    father_path = os.path.abspath(input_dir)
    print(father_path)
    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(['sh', 'dataset/Movie_Poster/eval.sh', father_path])

    if debug:
        source_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
        outpt_dir_base = os.path.join(cfg.output_dir, "Analysis", "eval_view", "Movie_Poster")
        if not os.path.exists(outpt_dir_base):
            mkdirs(outpt_dir_base)

        outpt_dir = os.path.join(outpt_dir_base, "{}_{}_{}".format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch))
        osmkdir(outpt_dir)
        fid_path1 = '{}/Eval_Movie_Poster.txt'.format(eval_dir)

        analysize_result(source_dir, fid_path1, outpt_dir, "Movie_Poster")

    print('End.')


def data_transfer_ICDAR(contours):
    cnts = list()
    for cont in contours:
        rect = cv2.minAreaRect(cont)
        if min(rect[1][0], rect[1][1]) <= 5:
            continue
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        # print(points.shape)
        # points = np.reshape(points, (4, 2))
        cnts.append(points)
    return cnts


def data_transfer_TD500(contours, res_file, img=None):
    with open(res_file, 'w') as f:
        for cont in contours:
            rect = cv2.minAreaRect(cont)
            if min(rect[1][0], rect[1][1]) <= 5:
                continue
            points = cv2.boxPoints(rect)
            box = np.int0(points)
            cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

            cx, cy = rect[0]
            w_, h_ = rect[1]
            angle = rect[2]
            mid_ = 0
            if angle > 45:
                angle = 90 - angle
                mid_ = w_;
                w_ = h_;
                h_ = mid_
            elif angle < -45:
                angle = 90 + angle
                mid_ = w_;
                w_ = h_;
                h_ = mid_
            angle = angle / 180 * 3.141592653589

            x_min = int(cx - w_ / 2)
            x_max = int(cx + w_ / 2)
            y_min = int(cy - h_ / 2)
            y_max = int(cy + h_ / 2)
            f.write('{},{},{},{},{}\r\n'.format(x_min, y_min, x_max, y_max, angle))

    return img

def data_transfer_Movie_Poster(contours, res_file, img=None):
    with open(res_file, 'w') as f:
        for cont in contours:#cont是预测出来的一个框的点,contours是全部框的全部点
            rect = cv2.minAreaRect(cont)#使用 cv2.minAreaRect 函数计算最小外接矩形。这个函数接受一个轮廓作为输入，并返回一个元组，包含矩形的中心坐标[0]、宽度、高度[1]和旋转角度[2]。
            if min(rect[1][0], rect[1][1]) <= 5:#检查最小外接矩形的宽度和高度的最小值是否小于或等于5。如果是，说明这个矩形太小，可能是噪声或者不感兴趣的区域，跳过这个轮廓。
                continue
            points = cv2.boxPoints(rect)#使用 cv2.boxPoints 函数从最小外接矩形中获取矩形的四个角点坐标。如果rect是((cx, cy), (w, h), angle)，其中((cx, cy)是矩形中心坐标，(w, h)是矩形的宽度和高度，angle是矩形的旋转角度，那么cv2.boxPoints(rect)将返回一个形状为(4, 2)的NumPy数组，包含四个角点的坐标。
            box = np.int0(points)#将浮点型的角点坐标转换为整数
            cv2.drawContours(img, [box], 0, (0, 255, 0), 3)#在原始图像上绘制最小外接矩形。[box] 是一个包含四个角点的列表，0 表示绘制第一个轮廓（即最小外接矩形），(0, 255, 0) 是绘制颜色（绿色），3 是线宽度。

            cx, cy = rect[0]
            w_, h_ = rect[1]
            angle = rect[2]
            mid_ = 0
            if angle > 45:
                angle = 90 - angle
                mid_ = w_;
                w_ = h_;
                h_ = mid_#交换宽度和高度的值。
            elif angle < -45:
                angle = 90 + angle
                mid_ = w_;
                w_ = h_;
                h_ = mid_
            angle = angle / 180 * 3.141592653589#交换宽度和高度的值。

            x_min = int(cx - w_ / 2)
            x_max = int(cx + w_ / 2)
            y_min = int(cy - h_ / 2)
            y_max = int(cy + h_ / 2)
            f.write('{},{},{},{},{}\r\n'.format(x_min, y_min, x_max, y_max, angle))

    return img


def data_transfer_MLT2017(contours, res_file):
    with open(res_file, 'w') as f:
        for cont in contours:
            rect = cv2.minAreaRect(cont)
            if min(rect[1][0], rect[1][1]) <= 5:
                 continue
            ploy_area = cv2.contourArea(cont)
            rect_area = rect[1][0]*rect[1][1]
            solidity = ploy_area/rect_area
            width = rect[1][0] - np.clip(rect[1][0] * (1-np.sqrt(solidity)), 0, 6)
            height = rect[1][1] - np.clip(rect[1][1] * (1-np.sqrt(solidity)), 0, 4)
            points = cv2.boxPoints((rect[0], (width, height), rect[2]))
            points = np.int0(points)
            p = np.reshape(points, -1)
            f.write('{},{},{},{},{},{},{},{},{}\r\n'
                    .format(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 1))



