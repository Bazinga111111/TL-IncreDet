import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('light weight.yaml')  # 此处填写教师模型的权重文件地址
    #model_t = YOLO(r'yolov8n.pt')  # 此处填写教师模型的权重文件地址
 
    #model_t.model.model[-1].set_Distillation = True  # 不用理会此处用于设置模型蒸馏

    #model_s = YOLO('light weight_GSConv.yaml')
    #model.load('D:/work/thesis/light weight/runs/detect/YOLOv8s_seq123_5epoch/weights/best.pt')
    model.load('YOLOv8s.pt')

    model.train(data='light weight.yaml', epochs = 50, batch = 1)
    #model_s.train(data='light weight_GSConv.yaml', epochs = 10)

    # model_s.train(data=r'C:\Users\Administrator\Desktop\Snu77\ultralytics-main\New_GC-DET\data.yaml',  #  将data后面替换你自己的数据集地址
    #             cache=False,
    #             imgsz=640,
    #             epochs=100,
    #             single_cls=False,  # 是否是单类别检测
    #             batch=1,
    #             close_mosaic=10,
    #             workers=0,
    #             device='0',
    #             optimizer='SGD',  # using SGD
    #             amp=True,  # 如果出现训练损失为Nan可以关闭amp
    #             project='runs/train',
    #             name='exp',
    #             model_t=model_t.model
    #             )

    #model.val()


 
