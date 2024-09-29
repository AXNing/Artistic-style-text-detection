
import re
import os
import numpy as np
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
from util.misc import norm2


class Movie_postertext(TextDataset):

    def __init__(self, data_root, is_training=True, ignore_list=None, load_memory=False, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_root = os.path.join(data_root, 'Train/img' if is_training else 'Test/img')
        #self.image_root = "/home/ubuntu/TextBPN-Plus-Plus-main/data/jijin/Test"

        self.annotation_root = os.path.join(data_root,'Train/gt' if is_training else 'Test/gt')
        self.image_list = os.listdir(self.image_root)
        self.annotation_list=os.listdir(self.annotation_root)
        p = re.compile('.rar|.txt')#表示匹配以 .rar 或 .txt 结尾的字符串
        # self.image_list = [x for x in self.image_list if not p.findall(x)]#通过列表推导式，将其中不包含 .rar 或 .txt 结尾的字符串保留下来
        self.image_list = [x for x in self.image_list ]#通过列表推导式，将其中不包含 .rar 或 .txt 结尾的字符串保留下来
        self.image_list=sorted(self.image_list)

        p = re.compile('(.jpg|.JPG|.PNG|.JPEG)')
        
        self.annotation_list = [x for x in self.annotation_list]#替换去除文件名中以 .jpg、.JPG、.PNG 或 .JPEG 结尾的后缀，并将结果存储在 self.annotation_lis
        self.annotation_list=sorted(self.annotation_list)

        # self.annotation_list = ['{}'.format(p.sub("", img_name)) for img_name in self.image_list]#替换去除文件名中以 .jpg、.JPG、.PNG 或 .JPEG 结尾的后缀，并将结果存储在 self.annotation_lis

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))#根据条件是否加载到内存，使用 self.load_img_gt 方法加载图像和相应注释，并将它们存储在 self.datas 列表中


    @staticmethod
    def parse_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path)#读取文件中的所有文本行，并返回一个包含这些行的列表
        polygons = []
        for line in lines:
            line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')#移除每行开头的 \ufeff 和 \xef\xbb\xbf，这可能是由于文件编码引起的
            gt = line.split(',')#将每行以逗号分隔的值拆分为一个列表，存储在 gt 中
            coordinates = list(map(float, gt[:-1]))  # 假设最后一个值是label，不是坐标
                # 将坐标值分组成 x 和 y
            xx = coordinates[::2]
            yy = coordinates[1::2]
            label_name = gt[-1]
            label = gt[-1].strip().replace("###", "#")#从 gt 中获取最后一个值，去除首尾空格，并将 "###" 替换为 "#"，然后存储在 label 中
            pts = np.stack([xx, yy]).T.astype(np.int32)#创建一个包含 x 和 y 坐标的 NumPy 数组，并将其转置为 (N, 2) 形状，然后将数据类型转换为 np.int32
            #np.stack 将输入数组的序列沿新轴堆叠。在这里，xx 和 yy 是两个一维数组，该函数将它们堆叠在一起，形成一个包含两行的二维数组，其中第一行是 xx 的值，第二行是 yy 的值。.T：这是 NumPy 中数组的转置操作，将数组的行和列交换。在这里，它将原本的两行变成了两列，使得每列包含了一个坐标点的 x 和 y 值。.astype(np.int32)：这一步将数组中的元素类型转换为 np.int32，确保坐标是整数类型。
            
            
            # d1 = norm2(pts[0] - pts[1])
            # d2 = norm2(pts[1] - pts[2])
            # d3 = norm2(pts[2] - pts[3])
            # d4 = norm2(pts[3] - pts[0])
            # if min([d1, d2, d3, d4]) < 2:
            #     continue
            polygons.append(TextInstance(pts, 'c', label))#创建一个 TextInstance 实例，该实例包含了文本框的坐标信息 (pts)、文本类型（'c'，可能表示文本类别）以及标签信息 (label)。这个实例被追加到 polygons 列表中。

        return polygons

    def load_img_gt(self, item):
        image_id = self.image_list[item]#从 self.image_list 中获取索引为 item 的图像文件的名称
        image_path = os.path.join(self.image_root, image_id)
        # Read image data
        image = pil_load_img(image_path)
        try:
            # Read annotation
            annotation_id = self.annotation_list[item]#从 self.annotation_list 中获取索引为 item 的注释文件的名称，并构建注释文件的完整路径
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            polygons = self.parse_txt(annotation_path)##获取标注点，将其转化为numpy数组
        except:
            polygons = None

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
        data["image_path"] = image_path
        return data

    def __getitem__(self, item):
        # image_id = self.image_list[item]
        # image_path = os.path.join(self.image_root, image_id)
        #
        # # Read image data
        # image = pil_load_img(image_path)
        #
        # try:
        #     # Read annotation
        #     annotation_id = self.annotation_list[item]
        #     annotation_path = os.path.join(self.annotation_root, annotation_id)
        #     polygons = self.parse_txt(annotation_path)
        # except:
        #     polygons = None

        if self.load_memory:
            data = self.datas[item]
        else:
            data = self.load_img_gt(item)

        if self.is_training:
            return self.get_training_data(data["image"], data["polygons"],
                                          image_id=data["image_id"], image_path=data["image_path"])
        else:
            return self.get_test_data(data["image"], data["polygons"],
                                      image_id=data["image_id"], image_path=data["image_path"])

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    from util.augmentation import BaseTransform, Augmentation

    import os
    import cv2
    from util.augmentation import Augmentation
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Movie_Poster_text(
        data_root='../data/Movie_Poster',
        is_training=False,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]
    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask = trainset[idx]
        img, train_mask, tr_mask = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)
        print(idx, img.shape)

        for i in range(tr_mask.shape[2]):
            cv2.imshow("tr_mask_{}".format(i),
                       cav.heatmap(np.array(tr_mask[:, :, i] * 255 / np.max(tr_mask[:, :, i]), dtype=np.uint8)))

        cv2.imshow('imgs', img)
        cv2.waitKey(0)
