# Task 2 拟合结果报告

## 1. 任务概述
- 视频：`resources/task_2.mp4`（960×720 60 fps，共 1440 帧）
- 目标：跟踪旋转的蓝色标记点，提取角速度 ω(t) 时间序列，并用正弦模型拟合。
- 输出：
  - `output_result.mp4` — 跟踪过程可视化
  - `fit_comparison.png` — 原始采样点 vs 拟合曲线
  - `angular_velocity.png` — 拟合出的角速度曲线
  - `residuals.png` — 残差曲线

## 2. 模型

角速度随时间的变化模型为带直流偏置的正弦振荡：

$$
\omega(t) = b + A \sin(\Omega t + \varphi)
$$

| 符号 | 物理含义 | 单位 |
|---|---|---|
| $b$ | 角速度直流分量（平均角速度） | rad/s |
| $A$ | 振荡幅值 | rad/s |
| $\Omega$ | 振荡角频率 | rad/s |
| $\varphi$ | 初相位 | rad |

## 3. 方法

### 3.1 角速度提取
1. **目标分割**：HSV 阈值提取蓝色标记（H∈[80,150], S∈[120,255], V∈[50,255]）→ 形态学开运算去噪 → `findContours` 取轮廓 → `minEnclosingCircle` 得目标中心 $(x_c, y_c)$。
2. **角度**：以图像中心 (480, 360) 为原点计算方位角 $\theta = \mathrm{atan2}(y_0 - y_c,\ x_c - x_0)$。
3. **角速度**：逐帧差分 $\omega = \Delta\theta / \Delta t$，并做相位解缠绕（把 $\Delta\theta$ 折叠回 $[-\pi,\pi]$）。
4. **降采样**：逐帧差分噪声大，取 `kSample = 7`（差分间隔 7/60 s），共 205 个采样点。

### 3.2 非线性最小二乘拟合（Ceres Solver）
- **残差**：$r_i = \omega_i - \big(b + A\sin(\Omega t_i + \varphi)\big)$
- **自动求导**：`ceres::AutoDiffCostFunction<SineResidual, 1, 4>`
- **求解器**：Levenberg–Marquardt 信赖域，线性求解器 `DENSE_QR`
- **参数块**：`p[4] = {b, A, Ω, φ}`，初值 `{1.3, 0.5, 1.6, 0.5}`


## 4. 拟合参数

| 参数 | 拟合值 |
|---|---|
| $b$ | 1.3501 rad/s |
| $A$ | 0.5490 rad/s |
| $\Omega$ | 1.6498 rad/s |
| $\varphi$ | 0.606 rad |


## 5. 误差指标

| 指标 | 公式 | 值 |
|---|---|---|
| 均方根误差 RMSE | $\sqrt{\dfrac{1}{N}\sum r_i^2}$ | 0.0123 rad/s |
| 决定系数 R² | $1 - \dfrac{\sum r_i^2}{\sum(\omega_i - \bar{\omega})^2}$ | 0.999 |

## 6. 结论
- R² = 0.999、RMSE 仅 0.0123 rad/s，正弦模型能很好地描述该旋转运动的角速度变化。

