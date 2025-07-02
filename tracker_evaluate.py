import os
import numpy as np
import matplotlib.pyplot as plt

def yolo_to_xywh(yolo_box, img_width, img_height):
    """
    将YOLO格式的框转为 [x_center, y_center, width, height] 格式
    :param yolo_box: YOLO框数据 [class, x_center, y_center, width, height]
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :return: 转换后的框 [x_center, y_center, width, height]
    """
    # YOLO中的坐标是相对的，我们需要转换为绝对的
    x_center = yolo_box[1] * img_width
    y_center = yolo_box[2] * img_height
    width = yolo_box[3] * img_width
    height = yolo_box[4] * img_height
    
    return [x_center, y_center, width, height]

def calculate_iou(box1, box2):
    """
    计算两个框之间的 IoU（交并比）
    :param box1: [x_center, y_center, width, height]
    :param box2: [x_center, y_center, width, height]
    :return: IoU值
    """
    # 计算框的边界
    box1 = box1[0]  # 获取第一个框数据
    box2 = box2[0]  # 获取第一个框数据
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0  # 无交集
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def calculate_center_error(box1, box2):
    """
    计算两个框之间的中心点误差（欧几里得距离）
    :param box1: [x_center, y_center, width, height]
    :param box2: [x_center, y_center, width, height]
    :return: 中心点误差
    """
    # 提取中心点坐标
    box1 = box1[0]  # 获取第一个框数据
    box2 = box2[0]  # 获取第一个框数据
    x1, y1 = box1[0], box1[1]
    x2, y2 = box2[0], box2[1]
    
    # 计算欧几里得距离
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def load_yolo_boxes_from_txt(txt_folder, img_width, img_height):
    """
    从文件夹中读取所有txt文件并解析为YOLO框格式
    :param txt_folder: 存储YOLO格式标注文件的文件夹路径
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :return: 所有YOLO框的列表，格式为 [[x_center, y_center, width, height], ...]
    """
    all_boxes = []
    
    # 获取所有txt文件路径
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        txt_path = os.path.join(txt_folder, txt_file)
        with open(txt_path, 'r') as f:
            boxes = []
            for line in f:
                yolo_box = list(map(float, line.strip().split()))
                boxes.append(yolo_to_xywh(yolo_box, img_width, img_height))
            all_boxes.append(boxes)
    
    return all_boxes

def evaluate_tracker(pred_boxes, gt_boxes, iou_thresholds=np.arange(0, 1.1, 0.1), center_thresholds=np.arange(0, 1.1, 0.1)):
    """
    评估目标跟踪的 Success 和 Precision
    :param pred_boxes: 预测的框列表，每一帧为一项 [x_center, y_center, width, height]
    :param gt_boxes: 真实框列表，每一帧为一项 [x_center, y_center, width, height]
    :param iou_thresholds: IoU阈值数组
    :param center_thresholds: 中心点误差阈值数组
    :return: Success 和 Precision 图
    """
    success = []
    precision = []
    
    # 计算 Success (基于 IoU)
    for iou_threshold in iou_thresholds:
        success_count = 0
        for pred, gt in zip(pred_boxes, gt_boxes):
            if calculate_iou(pred, gt) >= iou_threshold:
                success_count += 1
        success.append(success_count / len(gt_boxes))  # 成功率

    # 计算 Precision (基于中心点误差)
    for center_threshold in center_thresholds:
        precision_count = 0
        for pred, gt in zip(pred_boxes, gt_boxes):
            if calculate_center_error(pred, gt) <= center_threshold:
                precision_count += 1
        precision.append(precision_count / len(gt_boxes))  # 精度

    return success, precision

def plot_metrics(success, precision, iou_thresholds, center_thresholds):
    """
    绘制 Success 和 Precision 图
    :param success: Success列表
    :param precision: Precision列表
    :param iou_thresholds: IoU阈值数组
    :param center_thresholds: 中心点误差阈值数组
    """
    # 绘制 Success 图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(iou_thresholds, success, marker='o', color='b')
    plt.title('Success vs IoU Threshold')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Success Rate')
    plt.grid(True)
    
    # 绘制 Precision 图
    plt.subplot(1, 2, 2)
    plt.plot(center_thresholds, precision, marker='o', color='r')
    plt.title('Precision vs Center Point Error Threshold')
    plt.xlabel('Center Point Error Threshold')
    plt.ylabel('Precision Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 示例：假设图像宽度和高度为 1280 和 720
img_width = 1280
img_height = 720

# 假设指定路径分别为预测框和真实框数据
pred_folder = 'D:/work/thesis/video_data/tracker_data/labels'  # 预测框文件夹路径
gt_folder = 'D:/work/thesis/video_data/video1/labels'   # 真实框文件夹路径

# 加载YOLO标注数据
pred_boxes = load_yolo_boxes_from_txt(pred_folder, img_width, img_height)
gt_boxes = load_yolo_boxes_from_txt(gt_folder, img_width, img_height)

# 评估并绘制图表
iou_thresholds = np.arange(0, 1.1, 0.1)
center_thresholds = np.arange(0, 51, 5)
success, precision = evaluate_tracker(pred_boxes, gt_boxes, iou_thresholds, center_thresholds)
plot_metrics(success, precision, iou_thresholds, center_thresholds)