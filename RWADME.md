# Task1 图像处理基础 — 结果说明

基于 OpenCV 的图像处理练习：从测试图里分割出红色花瓣，并把常用 API（色彩空间、滤波、形态学、轮廓、绘图、几何变换）逐个跑一遍。

**环境**：Ubuntu 22.04 + g++ 11.4.0（C++17）+ OpenCV 4.5.4，无其他依赖。

---

## 一、构建

```bash
cmake -S . -B build
cmake --build build
```

## 二、运行

```bash
cd src/task1_image && ../../build/task1_image
cd src/task2_fit && ../../build/task2_fit
cd src/task3_windmill && ../../build/task3_windmill
cd src/task3_windmill && ../../build/task4_windmill
```
---

## 三、输入输出路径

| 任务 | 源码 | 输入 | 输出 |
|---|---|---|---|
| task1  | `src/task1_image/main.cpp` | `resources/test_image.jpg` | `result/task1_images/`（16 张 PNG） |
| task2 | `src/task2_fit/main.cpp` | `resources/task_2.mp4` | `result/task2_fit/`（1 个 mp4 + 3 张 PNG） |
| task3 | `src/task3_windmill/main.cpp` | `resources/task_3.mp4` | `result/task3_windmill/task3/`（2 个 mp4） |
| task4 | `src/task3_windmill/main2.cpp` | `resources/task_4.mp4` | `result/task3_windmill/task4/`（2 个 mp4） |

**输入素材**（`resources/`）

| 文件 | 规格 |
|---|---|
| `test_image.jpg` | 静态图：红郁金香 + 黄橙花瓣边 + 绿茎 + 灰背景 |
| `task_2.mp4` | 960×720 @60fps，1440 帧 |
| `task_3.mp4` | 1440×1080 @30fps，796 帧 |
| `task_4.mp4` | 1440×1080 @30fps，1800 帧 |

---

## 四、关键参数

### task1

```cpp
inRange(hsv, Scalar(0, 100, 80),   Scalar(10, 255, 255), maskLow);    // H 低段
inRange(hsv, Scalar(170, 100, 80), Scalar(179, 255, 255), maskHigh);  // H 高段
```

| 参数 | 值 | 含义 |
|---|---|---|
| H | `0~10` 与 `170~179` | 红色跨 0 度，必须分两段 |
| S 下限 | 100 | 饱和度 |
| V 下限 | 80 | 管亮度 |

### task2（Ceres 拟合 ω(t)）

| 参数 | 值 | 含义 |
|---|---|---|
| `kSample` | 7 | 角度序列降采样间隔 |
| `kCurvePts` | 600 | 画 ω(t) 曲线用的采样点数 |

### task3（风车单目标追踪）

| 常量 | 值 | 含义 |
|---|---|---|
| 圆度阈值 | `>= 0.8` | 轮廓分类 |
| 面积阈值 | `>= 300` | 滤噪点 |
| `MAX_SAME_D` | 150 px | 连续追踪时的位移门限 |
| `REAPPEAR_D` | 250 px | 短暂丢失后重现时的放宽门限 |
| `MAX_LOST` | 30 帧 | 丢失容忍 |
| `D_R_MAX` | 260 px | 到 R 标距离上限，超过判为异常帧、ID 保持不变 |
| 膨胀核 | `7×7` 椭圆 | 合并碎片，否则 R 标细结构过不了面积/圆度筛选 |

### task4（能量机关双目标追踪）

| 常量 | 值 | 含义 |
|---|---|---|
| 圆度阈值 | `>= 0.6` | 轮廓分类|
| 目标圆 | 面积 `>= 4000` 且 `nChild >= 2` | 实心大圆 |
| R 标候选 | 面积 400~1100 | — |
| `TRACK_D` | 150 px | 一直在追的槽位 |
| `RECOVER_D` | 280 px | 丢过的槽位 |
| `TOLERATE` | 45 帧 | 容忍帧数|
| `D_R_MIN / D_R_MAX` | 40 / 250 px | 目标到 R 标的环带范围 |
| 编号分配 | `nextId++` |下一个圆的id |

---

## 五、任务 1 分析

目标：从测试图里把红色花分割出来。素材是红郁金香 + 黄橙花瓣边 + 绿茎 + 灰背景，难点在于红色和黄色在 HSV 色环上是**相邻**的，阈值定偏就会吃进花瓣边缘。

### 1. 色调 H 上限的取舍

红色跨 0 度，所以 H 要分 `0~10` 和 `170~179` 两段。考虑**低段的上限**——放太宽就把橙黄花瓣边也吃进来了

### 3. 滤波顺序
用 `inRange` 切出二值图，再用腐蚀/膨胀/开闭运算处理噪点和孔洞。
---

## 六、结果索引

全部结果在 `result/task1_images/`，共 16 张 PNG：

| 文件 | 内容 |
|---|---|
| `gray.png` | 灰度图 |
| `hsv_h.png` / `hsv_s.png` / `hsv_v.png` | HSV 三个通道分离 |
| `mean.png` / `gaussian.png` / `median.png` | 均值 / 高斯 / 中值滤波对比 |
| `red_mask.png` | **红色二值掩膜**（核心结果） |
| `erode.png` / `dilate.png` / `open.png` / `close.png` | 腐蚀 / 膨胀 / 开运算 / 闭运算 |
| `contours_boxes.png` | 轮廓 + 外接矩形 + 面积标注 |
| `draw_shapes.png` | 绘图 API 练习（圆 / 矩形 / 文字） |
| `rotate35.png` / `crop_topleft.png` | 旋转 35° / 裁剪左上角 |
