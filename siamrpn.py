#https://zhuanlan.zhihu.com/p/161983646
from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import namedtuple
from got10k.trackers import Tracker


class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()   #调用父类的初始化方法，这里的父类为nn.Module
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(   #self.feature是一个特征提取子网络,它的作用是从输入图像中提取相关的特征信息(孪生网络，参数相同)
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))
        
        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)   #self.conv_reg_z是用于回归位置的卷积层。它将feature提取到的模板的特征图进行卷积，输出模板的位置特征信息
        self.conv_reg_x = nn.Conv2d(512, 512, 3)                       #self.conv_reg_x是用于回归搜索区域的卷积层。它将输入的搜索区域特征图进行卷积，输出预测搜索区域的相关信息
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)   #self.conv_cls_z是用于分类目标的卷积层。它将feature提取到的模板的特征图进行卷积，输出模板的类别特征信息
        self.conv_cls_x = nn.Conv2d(512, 512, 3)                       #self.conv_cls_x是用于分类搜索区域的卷积层。它将输入的搜索区域特征图进行卷积，输出预测搜索区域分类的相关信息
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1) #self.adjust_reg是一个调整回归结果的卷积层。它用于进一步优化回归结果，调整预测的目标位置

    def forward(self, z, x):   #一次完整的前向推理过程
        return self.inference(x, **self.learn(z))

    def learn(self, z):
        '''
        SiamRPN网络时的模板卷积核参数学习过程,给定输入的模板图像,输出用于回归分支和分类分支的卷积核参数
        z: 输入的模板图像
        '''
        z = self.feature(z)               #提取特征子网络
        kernel_reg = self.conv_reg_z(z)   #得到模板回归分支的卷积核参数
        kernel_cls = self.conv_cls_z(z)   #得到模板分类分支的卷积核参数

        #将回归和分类卷积核参数进行reshape操作
        k = kernel_reg.size()[-1]                                      #获取回归卷积核参数的大小，即卷积核的宽高
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k)   #把回归卷积核参数reshape为4倍锚点数目乘上512通道数，大小为k*k的4维张量
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k)   #把分类卷积核参数reshape为2倍锚点数目乘上512通道数，大小为k*k的4维张量

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls): 
        '''
        SiamRPN网络的匹配过程,给定输入搜索图像和预先学习到的模板的卷积核参数,输出目标的位置和分类信息
        x: 输入的图像
        kernel_reg: 用于回归目标位置的卷积核参数
        kernel_cls: 用于分类目标的卷积核参数       
        '''  
        x = self.feature(x)          #提取特征子网络
        x_reg = self.conv_reg_x(x)   #回归子网络
        x_cls = self.conv_cls_x(x)   #分类子网络
        
        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))   #使用回归卷积核参数kernel_reg对回归结果x_reg进行卷积，并通过self.adjust_reg层调整回归结果，得到目标的位置预测out_reg
        out_cls = F.conv2d(x_cls, kernel_cls)                    #使用分类卷积核参数kernel_cls对分类结果x_cls进行卷积，得到目标的分类预测out_cls

        return out_reg, out_cls


class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(name='SiamRPN', is_deterministic=True)   #调用父类的初始化
        self.parse_args(**kargs)

        # 检查是否有可用的GPU设备，并选择使用cuda还是cpu进行计算
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # 加载SiamRPN模型并将其移动到所选的设备
        self.net = SiamRPN()                  # 创建SiamRPN模型对象为net
        if net_path is not None:              # 如果给定模型的路径，就用torch.load加载模型
            self.net.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)   # 将模型移动到指定的设备(CUDA或CPU)

    def parse_args(self, **kargs):
        self.cfg = {                        # 定义了一些默认的参数值
            'exemplar_sz': 127,                # 模板的大小
            'instance_sz': 271,                # 候选区域的大小
            'total_stride': 8,                 # anchor生成的步长
            'context': 0.5,
            'ratios': [0.33, 0.5, 1, 2, 3],
            'scales': [8,],
            'penalty_k': 0.055,
            'window_influence': 0.42,
            'lr': 0.295}

        for key, val in kargs.items():        # 更新增加的参数
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)   # 双星操作符用于解包元组内数据，此处生成具名元组GenericDict，方便获取参数的值

    def init(self, image, box):
        image = np.asarray(image)

        # 将目标边界框的坐标转换为0索引并基于中心点的形式
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]   # box即[y, x, h, w]。其中，y和x是目标中心点的坐标，h和w是边界框的高度和宽度

        # 如果目标太小，则使用更大的搜索区域，获取更多的上下文信息
        if np.prod(self.target_sz) / np.prod(image.shape[:2]) < 0.004:
            self.cfg = self.cfg._replace(instance_sz=287)

        # 创建anchors
        self.response_sz = (self.cfg.instance_sz - self.cfg.exemplar_sz) // self.cfg.total_stride + 1   # '//'返回除法运算的整数部分，这里计算一个方向上需要生成的anchor的数量
        self.anchors = self._create_anchors(self.response_sz)

        # 生成汉宁窗
        self.hann_window = np.outer(   # np.outer一般与np.hanning搭配在一起，用来生成高斯矩阵
            np.hanning(self.response_sz),
            np.hanning(self.response_sz))
        self.hann_window = np.tile(    # 每个位置复制对应anchor数量份
            self.hann_window.flatten(),
            len(self.cfg.ratios) * len(self.cfg.scales))

        # 计算搜索区域的大小
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))                # np.prod()计算所有元素的乘积，此处计算考虑了context增益后的模板中的search size
        self.x_sz = self.z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz   #此处计算考虑了context增益后的输入图像中的search size(等比放大)

        # 生成模板图像
        self.avg_color = np.mean(image, axis=(0, 1)) 
        exemplar_image = self._crop_and_resize(   # 对输入图像进行裁剪和缩放操作，得到模板图像
            image, self.center, self.z_sz, self.cfg.exemplar_sz, self.avg_color)

        # 使用PyTorch对模板图像进行分类和回归处理
        exemplar_image = torch.from_numpy(exemplar_image).to(    # 将NumPy数组转换为PyTorch张量
            self.device).permute([2, 0, 1]).unsqueeze(0).float() # 用.to(self.device)将张量移动到指定的计算设备（如GPU）
                                                                 # 接着使用.permute([2, 0, 1])对通道进行重新排列，将通道维度放到第二个位置
                                                                 # 最后使用.unsqueeze(0)在第0维上增加一个维度，以匹配网络的输入要求，并将其转换为浮点类型数据
        with torch.set_grad_enabled(False):
            self.net.eval()                                                     #开启评估模式，这里相当于self.net.train(False)
            self.kernel_reg, self.kernel_cls = self.net.learn(exemplar_image)   #获取回归、分类卷积核参数

    def update(self, image):
        image = np.asarray(image)
        
        # 裁剪出候选区域图像，并缩放为指定大小
        instance_image = self._crop_and_resize(
            image, self.center, self.x_sz,
            self.cfg.instance_sz, self.avg_color)

        # 使用PyTorch对候选区域图像进行分类和回归处理(得到位置、分类预测结果out_reg, out_cls)
        instance_image = torch.from_numpy(instance_image).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            out_reg, out_cls = self.net.inference(
                instance_image, self.kernel_reg, self.kernel_cls)
        
        # 计算偏移量(此处为回归分支的结果)
        offsets = out_reg.permute(
            1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
        offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
        offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
        offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]
        offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]

        # 尺度和比例惩罚
        penalty = self._create_penalty(self.target_sz, offsets)

        # 计算响应图(此处为分类分支的结果)
        response = F.softmax(out_cls.permute(
            1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
        response = response * penalty
        response = (1 - self.cfg.window_influence) * response + self.cfg.window_influence * self.hann_window
        # 标记最可能的目标位置
        best_id = np.argmax(response)   #返回响应图中最大值的索引
        offset = offsets[:, best_id] * self.z_sz / self.cfg.exemplar_sz

        # 对候选区域进行微调
        # 更新中心点
        self.center += offset[:2][::-1]
        self.center = np.clip(self.center, 0, image.shape[:2])
        # 更新尺度
        lr = response[best_id] * self.cfg.lr
        self.target_sz = (1 - lr) * self.target_sz + lr * offset[2:][::-1]
        self.target_sz = np.clip(self.target_sz, 10, image.shape[:2])

        # 更新模板和候选区大小
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = 1.15*np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz

        # 根据更新后的目标中心点和目标大小，计算目标的边界框坐标
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def _create_anchors(self, response_sz):   # 生成anchor
        anchor_num = len(self.cfg.ratios) * len(self.cfg.scales)  # 确定单一位置需要生成的anchor数量
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)     # 创建一个形状为(anchor_num, 4)的全零数组 anchors 来存储anchor的坐标信息

        size = self.cfg.total_stride * self.cfg.total_stride
        ind = 0
        for ratio in self.cfg.ratios:                             # 遍历不同的ration
            w = int(np.sqrt(size / ratio))                        # 先根据步长和比例关系来确定anchor的宽度
            h = int(w * ratio)                                    # 再利用宽度和比例关系确定anchor的高度
            for scale in self.cfg.scales:                         # 遍历不同的scales
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale
                anchors[ind, 3] = h * scale
                ind += 1
        anchors = np.tile(                                       # np.tile(A，reps)表示对输入A进行重复构造,此处相当于对整张图片生成anchor
            anchors, response_sz * response_sz).reshape((-1, 4))

        begin = -(response_sz // 2) * self.cfg.total_stride
        xs, ys = np.meshgrid(                                    # 计算每个anchor的中心坐标
            begin + self.cfg.total_stride * np.arange(response_sz),
            begin + self.cfg.total_stride * np.arange(response_sz))
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)

        return anchors

    def _create_penalty(self, target_sz, offsets):

        def padded_size(w, h):   # 用以计算考虑context之后的图像大小
            context = self.cfg.context * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):     # 返回当前比例系数和自身倒数两者中的最大值
            return np.maximum(r, 1 / r)
        
        src_sz = padded_size(*(target_sz * self.cfg.exemplar_sz / self.z_sz))   # 根据当前目标大小、模板大小和实例大小，计算源图像的大小
        dst_sz = padded_size(offsets[2], offsets[3])                            # 根据目标位置偏移量，计算目标位置的图像大小
        change_sz = larger_ratio(dst_sz / src_sz)                               # 计算尺度变化系数

        src_ratio = target_sz[1] / target_sz[0]                                 # 计算源图像的宽高比例
        dst_ratio = offsets[2] / offsets[3]                                     # 计算目标位置的宽高比例
        change_ratio = larger_ratio(dst_ratio / src_ratio)                      # 计算比例变化系数

        penalty = np.exp(-(change_ratio * change_sz - 1) * self.cfg.penalty_k)  # 根据尺度变化系数和比例变化系数，计算惩罚系数

        return penalty

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        '''
        此处为裁剪缩放函数
        image为原始图像
        center为目标中心点
        size参数传入的是图像输入大小
        out_size参数传入的是图像输出大小
        pad_color参数传入的是avg_color, 即图像的颜色平均值
        '''

        # 将box坐标(基于中心点)转为corners坐标(基于角坐标)
        size = round(size)           # round()功能为四舍五入
        corners = np.concatenate((   # np.concatenate()功能为拼接
            np.round(center - (size - 1) / 2),           # 要裁剪区域的左上角坐标
            np.round(center - (size - 1) / 2) + size))   # 要裁剪区域的右下角坐标
        corners = np.round(corners).astype(int)

        # 检查是否需要对图像进行padding
        pads = np.concatenate((-corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(   # cv2.copyMakeBorder函数在图像周围padding，填充颜色为pad_color
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # 裁剪图像
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # 调整至指定的输出大小
        patch = cv2.resize(patch, (out_size, out_size))

        return patch
