滑动窗口滤波器（Sliding-window Filter / 2D 卷积）的本质，是用一个小型权重矩阵（核 Kernel）在图像上逐像素滑动，通过局部区域像素与核权重的加权求和（点积）来提取空间特征或消除噪点。

图像在计算机中只是二维灰度矩阵 $I(x,y)$。卷积核里的数值分布决定了滤波器的功能：**权重全为正且和为 1 时，执行的是局部平均（平滑）；权重有正有负且和为 0 时，执行的是离散求导（检测变化与结构）。**

---

## 一、 6 种滤波器的发明逻辑与原理拆解

### 1. 模糊滤波 (Blurring Filter)

* **怎么想到的**：图像中的噪声（如高斯噪声、随机噪点）通常表现为单个像素亮度的骤变。要消除这种不自然的突变，最直观的想法就是“少数服从多数”，让周围邻居像素的平均值来平抑突变。
* **为什么可行**：
* **均值滤波**：取 $N \times N$ 窗口内所有像素的算术平均（权重均等为 $1/N^2$），相当于低通滤波器，滤除高频噪声。
* **高斯滤波**：考虑到距离中心越近的像素相关性越高，采用二维高斯分布 $G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$ 分配权重，越靠近中心的像素权重越大。


* **权重形状**：中心高、四周低的平滑山峰形状（或全平的方块）。
* **实际应用**：图像去噪、背景虚化、图像金字塔下采样前的抗锯齿处理。

---

### 2. 图像梯度 (Image Gradients: N4 / N8 邻域)

* **怎么想到的**：在微积分中，连续函数的导数定义为 $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$。离散图像的最小间隔 $h=1$，因此图像的“求导”直接简化为**邻近像素做差分**。
* **为什么可行**：
* **N4 邻域（4-正交邻域）**：仅计算上下左右 4 个方向的差分。如水平方向差分核为 $\begin{bmatrix} -1 & 0 & 1 \end{bmatrix}$，对应的导数估计为 $I_x \approx I(x+1, y) - I(x-1, y)$。
* **N8 邻域（8-全方向邻域）**：考虑到对角线方向像素的影响，在做差分的同时加入正交方向的平滑，避免单个像素噪点对梯度的干扰。


* **权重形状**：正负对称，两侧取值相反（如 $[-1, 0, 1]$），整体权重之和为 0。
* **实际应用**：计算图像边缘方向、计算光流场、构建图像特征描述子（如 SIFT/HOG 中的梯度直方图）。

---

### 3. 边缘检测 (Edge Detection - 如 Sobel 算子)

* **怎么想到的**：图像的“边缘”即为亮度发生剧烈变化的边界（即一阶导数/梯度幅值 $\vert{}\nabla I\vert{} = \sqrt{I_x^2 + I_y^2}$ 的极值点）。但直接做简单差分对高频噪声极其敏感。前人便想到：**将“一维差分”与“一维平滑”组合成二维算子**。
* **为什么可行**：以水平 Sobel 算子 $S_x$ 为例，它可以分解为两个一维向量的张量积：

$$S_x = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \begin{bmatrix} -1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$


* $\begin{bmatrix} -1 & 0 & 1 \end{bmatrix}$ 在 $x$ 方向做**差分**（求导检测边缘）。
* $\begin{bmatrix} 1 & 2 & 1 \end{bmatrix}^T$ 在 $y$ 方向做**高斯平滑**（压制噪声）。


* **权重形状**：一侧全为正，一侧全为负，中间为 0。
* **实际应用**：工业零件轮廓提取、车道线检测、医学图像分割。

---

### 4. 角点检测 (Corner Detection - 窗口局部特征分析)

* **怎么想到的**：分析滑动窗口在不同区域移动时的亮度变化：
1. **平坦区**：窗口向任何方向移动，内部亮度都不变；
2. **边缘区**：窗口沿边缘方向移动不变，垂直于边缘移动剧烈变化；
3. **角点区**：窗口**向任何方向移动，内部亮度都会发生剧烈变化**。


* **为什么可行**：构建结构张量（Structure Tensor / 二阶矩矩阵）$M$：

$$M = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$



通过滑动窗口 $w(x,y)$ 统计局部区域梯度的二元协方差。计算 $M$ 的两个特征值 $\lambda_1, \lambda_2$：若两个特征值都很高，说明在二维方向上均存在强梯度变化，该窗口中心即为角点（如 Harris 角点检测）。
* **权重形状**：矩阵核本身通常采用高斯函数作为窗口权重 $w(x,y)$。
* **实际应用**：图像拼接（Stitching）、三维重建（SFM）、视觉 SLAM 中的特征点跟踪。

---

### 5. 高斯拉普拉斯 (Laplacian of Gaussian, LoG)

* **怎么想到的**：一阶导数求极值找边缘不够精准；二阶导数 $\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$ 的过零点（Zero-crossing）能精准定位边缘中心线。但二阶导对噪声的放大非常严重。
* **为什么可以合并计算**：卷积与求导均为**线性时不变（LTI）操作**，满足交换律与结合律：

$$\nabla^2 (G * I) = (\nabla^2 G) * I$$



不必先做高斯平滑再求二阶导，而是**直接对高斯函数求二阶导得到解析算子 $\nabla^2 G$**，用这一个算子一步到位完成“平滑 + 二阶导极值点检测”。
* **权重形状**：经典的“墨西哥帽（Mexican Hat）”结构——中心为深坑（或突起），四周环绕相反符号的包络，总和为 0。
* **实际应用**：斑点检测（Blob Detection）、尺度空间极值提取（SIFT 算法的基础）。

---

### 6. Gabor 滤波器组 (Gabor Filter Bank)

* **怎么想到的**：生物学研究发现，哺乳动物大脑皮层 V1 区的简单细胞对视觉信号的选择具有方向性和空间频率选择性（即眼睛对某种特定角度、某种粗细的纹理特别敏感）。
* **为什么可行**：Gabor 滤波器由一个复正弦平面波（频率与方向选择器）**与一个**高斯包络（空间局域限制器）相乘得到：

$$g(x,y;\gamma,\theta,\psi,\sigma,\lambda) = \exp\left(-\frac{x'^2+\gamma^2y'^2}{2\sigma^2}\right) \cos\left(2\pi\frac{x'}{\lambda}+\psi\right)$$



其中 $x' = x\cos\theta + y\sin\theta$。
* 正弦波产生指定角度 $\theta$ 和频段 $\lambda$ 的干涉条纹（共振匹配特定纹理）；
* 高斯包络确保该特征检测只限制在局部区域内。
* **滤波器组（Bank）**：组合不同角度（如 $0^\circ, 45^\circ, 90^\circ, 135^\circ$）和不同尺度/频率的核，构成一组完备的纹理分析器。


* **权重形状**：具有方向性的多周期平行条纹，且幅值随距中心的距离按高斯衰减。
* **实际应用**：人脸识别、虹膜识别、纹理分类与指纹增强。

---

## 二、 NumPy 底层滑动窗口实现逻辑

为完成作业中的代码要求，不能直接调用 `cv2.filter2D`，需通过 NumPy 实现通用的 2D 滑动窗口卷积（带 Padding 保护）：

```python
import numpy as np
import matplotlib.pyplot as plt

def conv2d_sliding_window(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    使用 NumPy 底层逻辑实现的二维滑动窗口滤波器
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    # 计算填充边界 (Same Padding)
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # 对图像边界进行边缘复制填充
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=np.float64)
    
    # 滑动窗口遍历
    for i in range(img_h):
        for j in range(img_w):
            # 提取当前滑动窗口覆盖的图像子区域
            window = padded_img[i : i + k_h, j : j + k_w]
            # 窗口与核逐元素相乘并求和（点积）
            output[i, j] = np.sum(window * kernel)
            
    return output

# 常用核构建示例：
# 1. 高斯模糊核 (Blurring)
def get_gaussian_kernel(ksize=5, sigma=1.0):
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

# 2. Sobel 边缘检测核 (Edge Detection)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

# 3. 高斯拉普拉斯核 (LoG)
def get_log_kernel(ksize=5, sigma=1.0):
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    x, y = np.meshgrid(ax, ax)
    norm = (x**2 + y**2) / (2 * sigma**2)
    kernel = -1.0 / (np.pi * sigma**4) * (1.0 - norm) * np.exp(-norm)
    return kernel - np.mean(kernel) # 确保核权重的均值为0

```

在作业报告中画出核权重图（Plot of filter weights）时，可以直接对上述 `kernel` 数组使用 `plt.imshow(kernel, cmap='coolwarm')` 并加上 `plt.colorbar()`，就能清晰展现每个滤波器的正负分布与权重要点。