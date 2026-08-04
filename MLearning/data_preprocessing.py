import cv2
import numpy as np
import torch

def preprocess_image(img_path, target_size=(640, 640)):
    # 1. OpenCV 读取图片 -> 默认是 BGR, 形状是 [H, W, C]
    img = cv2.imread(img_path)
    
    # 2. 色彩空间转换: BGR -> RGB (因为深度学习模型多在 RGB 下训练)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. 尺寸缩放: 缩放到模型要求的输入尺寸 (例如 YOLOv5 默认 640x640)
    # 注意：OpenCV 的 resize 接收的参数是 (宽, 高) 即 (W, H)，与矩阵相反
    img_resized = cv2.resize(img_rgb, target_size)
    
    # 4. 转换成 PyTorch 张量，并转换数据类型为 float32
    tensor = torch.from_numpy(img_resized).float()
    
    # 5. 归一化: 从 [0, 255] 映射到 [0.0, 1.0] 
    tensor /= 255.0
    
    # 6. 维度置换 (Permute): 从 [H, W, C] 变成 [C, H, W]
    tensor = tensor.permute(2, 0, 1)
    
    # 7. 增加批次维度 (Batch Dimension): 从 [C, H, W] 变成 [1, C, H, W]
    # 因为神经网络期望接收一个 batch 的数据
    tensor = tensor.unsqueeze(0)
    
    return tensor, img # 返回处理后的张量和原图（用于后续画框）

def xywh_to_xyxy(x):
    """
    将 [x_center, y_center, w, h] 转换为 [xmin, ymin, xmax, ymax]
    支持 PyTorch 张量
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # xmin = x_center - w/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # ymin = y_center - h/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # xmax = x_center + w/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # ymax = y_center + h/2
    return y



