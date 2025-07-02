import cv2
import os

from siamrpn_m2 import TrackerSiamRPN

# 加载预训练的SiamRPN模型
net_path = 'siamrpn_model.pth'  # 模型路径
tracker = TrackerSiamRPN(net_path)

def xywh_to_yolo_format(image_width, image_height, bbox):
    """
    将 xywh 格式的 bbox 转换为 YOLO 格式.
    
    参数:
    - image_width: 图像的宽度
    - image_height: 图像的高度
    - bbox: (x, y, w, h) 边界框的 xywh 格式

    返回:
    - 转换后的 YOLO 格式的边界框 (class_id, x_center, y_center, width, height)
    """
    x, y, w, h = bbox
    # 计算 YOLO 格式中的各项参数（相对于图像大小的比例）
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    width = w / image_width
    height = h / image_height
    return f"0 {x_center} {y_center} {width} {height}"  # 假设只有一个类别 class_id=0

# 读取文件夹中的所有图片
def read_images_from_folder(folder_path):
    image_files = sorted(os.listdir(folder_path))
    image_paths = [os.path.join(folder_path, filename) for filename in image_files if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return image_paths

# 主要跟踪处理函数
def track_images(images_path, tracker, bounding_box):

    # 开始按顺序处理所有图片
    for i, image_path in enumerate(images_path):
        frame = cv2.imread(image_path)

        # 在当前帧上运行跟踪器
        bbox, isGood = tracker.update(frame)
        print(f"目标框: {bbox}, 完整性: {isGood}")
        yolo_format = xywh_to_yolo_format(1280, 720, bbox)
        label_name = "{:05d}.txt".format(i)
        label_filename = os.path.join(cropped_label_path + '/' + label_name)           
        image_name = "{:05d}.jpg".format(i)
        image_filename = os.path.join(cropped_image_path + '/' + image_name)
        with open(label_filename, 'w') as f:
            f.write(yolo_format)
        cv2.imwrite(image_filename, frame)  # 保存图像

        # 在当前帧上绘制边界框
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Frame', frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 清理并关闭窗口
    cv2.destroyAllWindows()

# 示例：设置图像文件夹路径和跟踪算法类型
video_number = 1
image_folder = 'D:/work/thesis/video_data/video' + str(video_number) + '/images'  # 替换为你的图片文件夹路径

cropped_label_path = f'D:/work/thesis/video_data/tracker_data/labels'
cropped_image_path = f'D:/work/thesis/video_data/tracker_data/images'
# 检查并创建标签文件夹路径
if not os.path.exists(cropped_label_path):
    os.makedirs(cropped_label_path)

# 检查并创建图像文件夹路径
if not os.path.exists(cropped_image_path):
    os.makedirs(cropped_image_path)

# 获取所有图片的路径
images_path = read_images_from_folder(image_folder)
# 读取第一张图片并初始化跟踪器
first_image = cv2.imread(images_path[0])
cv2.imshow('Frame', first_image)
bounding_box = cv2.selectROI('Frame', first_image, False)

# 初始化跟踪器
tracker.init(first_image, bounding_box)

# 可以传入一个初始化的 bounding_box（例如，通过手动标注或指定），否则会提示用户选择
track_images(images_path, tracker, bounding_box)