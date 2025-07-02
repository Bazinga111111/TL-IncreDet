import cv2
import os
#将视频文件中的每5帧提取并保存为图像文件
def video_to_image_sequence(video_path, output_dir):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    
    # 检查视频文件是否成功打开
    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # 获取视频的总帧数
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}")
    
    # 创建输出目录（如果不存在的话）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化帧计数器
    frame_count = 0
    number = 0

    while True:
        ret, frame = video_capture.read()  # 读取一帧

        if number % 5 == 0:
            if ret:
                # 构造输出图片的文件名
                output_path = os.path.join(output_dir, f"{frame_count:05d}.jpg")
                
                # 将图片保存为 JPG 格式
                cv2.imwrite(output_path, frame)
                frame_count += 1
                print(f"Processing frame {frame_count}/{int(total_frames/5)}...", end="\r")
            else:
                break
        number += 1
    
    # 释放视频文件资源
    video_capture.release()
    print("\nVideo to images conversion completed!")

# 示例：使用视频路径和输出路径
video_path = 'D:/work/thesis/video_data/video5.mp4'  # 请替换为你的视频文件路径
output_dir = 'D:/work/thesis/video_data/video5'      # 请替换为你希望保存图片的路径

video_to_image_sequence(video_path, output_dir)