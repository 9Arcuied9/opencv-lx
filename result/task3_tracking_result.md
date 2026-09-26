#  Task3 / Task4 结果说明



| | Task3 | Task4 |
|---|---|---|
| 源码 | `src/common/task3_windmill/main.cpp` | `src/common/task3_windmill/main2.cpp` |
| 输入 | `resources/task_3.mp4` | `resources/task_4.mp4` |
| 目标数 | 单个 | 最多两个（双槽位身份绑定） |
| 输出 | `result/task3_windmill/task3/` | `result/task3_windmill/task4/` |


---

## 一、检测方法（Task3 / Task4 共用）

### 1. 颜色分割（二值化）

能量机关的装甲板是橙红色，用 HSV 空间做分割。因为色调 H 跨越 0 度，必须分成两段再用按位或合并：

```cpp
cvtColor(frame, hsv, COLOR_BGR2HSV);
inRange(hsv, Scalar(0, 40, 50),   Scalar(10, 255, 255), mask1);   // 0~10 度
inRange(hsv, Scalar(170, 40, 50), Scalar(179, 255, 255), mask2);  // 170~179 度
bitwise_or(mask1, mask2, mask);
```

- 饱和度下限 40、亮度下限 50：只要颜色够"浓"就算目标，亮度下限放得低，暗部的橙红也能留住。
- 

### 2. 形态学膨胀

```cpp
Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
dilate(mask, mask, kernel);
```

- 7×7 的椭圆核把断开的笔画连成一个整体，后续筛选才成立。

### 3. 轮廓提取与初筛

```cpp
findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
```

用 `RETR_TREE` 是为了拿到**层级信息**，后面要靠子轮廓数量区分三类目标。初筛三条：

| 条件 | 作用 |
|---|---|
| `hierarchy[i][3] == -1` | 只看最外层轮廓 |
| `area >= 300` | 滤掉噪点和碎片 |
| `圆度 = 4π·面积 / 周长² >= 0.8` | 只要圆形的，正圆为 1 |

### 4. 按层级分类

对通过初筛的圆，统计它的**直接子轮廓数** `nChild`：

| nChild | 含义 | 用途 |
|---|---|---|
| `>= 2` | 目标扇叶圆（同心环 + 中心亮点） | 待追踪目标 |
| `== 1` | 被击中的空心环（只有一个内孔） |  |
| `== 0` | 实心小块（R 标、方向箭头、臂上黄条） | 其中面积 ≤1200 的作 R 标候选 |

### 5. 定位风车中心与 R 标

- **中心 hub**：所有 `nChild >= 1` 的圆心的平均值。扇叶在圆周上对称分布，它们的几何中心落在转轴上。
- **R 标**：在"非圆轮廓"（`圆度 < 0.8` 且 `面积 <= 1200`）里，取**离 hub 最近**的那个。R 标是风车中心那个带"R"字的徽标。

### 6. 绘制

目标圆心与 R 标中心之间连一条红线，R 标中心画蓝点 + 蓝十字，目标由 ID 追踪模块画绿圈。


## 二、Task3：单目标锁定与重选规则


### 1. 候选选取

每帧在**所有 `nChild >= 2` 的圆**里找离 `tPos` （目标最后已知位置）最近的那个，距离记为 `minD`。

### 2. 判定

```cpp
if (bestIdx >= 0) {                                   // 本帧检到了候选
    if (tId < 0)                    tId = nextId++;   // ① 首次锁定
    else if (farFromR)              /* ID 不变 */     // ② 异常帧
    else if (minD > sameD)          tId = nextId++;   // ③ 位移不合适 -> 换圆
    else if (frameNo - lastSeen > MAX_LOST) tId = nextId++;  // ④ 丢失超时 -> 换圆
    更新 tPos / tR / lastSeen，画绿圈 + "ID:n" + "TRACK"
} else if (tId >= 0) {                                // 本帧没检到
    在原位置画红字 "ID:n" + "LOST"（ID 保留，等目标回来）
}
```

**② 异常帧短路**：`dR = 目标到 R 标的距离`，若 `dR > 240px` 说明本帧的 R 标定位是误检（画面底部有橙色指示灯会被当成 R 标），此时距离判据不可信，直接判为"同一个圆，ID 不变"。

**③ 位移门限**：连续追踪时用 150px；如果目标上一帧刚丢过（`frameNo - lastSeen > 1`），说明它消失了几帧才回来、位置会飘

**④ 丢失超时**：连续 30 帧（约 1 秒）没检到，换新 ID。



## 三、Task4：双目标槽位匹配规则

Task4 的检测部分与 Task3 完全相同（见第一节），区别只在身份绑定：用**两个槽位**，每帧决定每个槽位认领哪个圆。

### 1. 槽位记录

```cpp
struct Slot {
    Point2f pos;        // 目标圆的位置
    float   radius;     // 半径
    int     id;         // 身份编号; -1 = 空槽位
    bool    tracking;   // 本帧有没有认到自己的圆
    int     lostFrames; // 容忍帧数
};
Slot slots[2];
int nextId = 1;       
```

### 2. 每帧三步

** 匹配**

- 场上**只剩一个圆**：先算它到两个槽位的距离
  - 两个都超过门限 → 判为“不是相同的圆”，分配新 ID
  - 有一个够近 → 判为“同一个圆”：**离得更近的那个槽位继承**（ID 不变，只更新位置/半径），另一个槽位记 loss
- 场上**不止一个圆**：每个槽位各自取离自己最近、且不超过门限的那个圆；两个槽位抢同一个圆时，谁更近谁继承

**记 loss**

没认到自己的圆的槽位：**位置和 ID 都不动**，只把 `lostFrames` 加 1；超过 `TOLERATE` 才把记录清空（`id = -1`，变回空槽位）。

**新圆**

没被任何槽位认领的圆 = 新目标：空槽位优先，否则挤掉丢得最久的槽位，然后分配 `nextId++`。

### 3. 两级距离门限

门限按槽位**上一帧的状态**选取：

```cpp
float lim = slots[i].tracking ? TRACK_D : RECOVER_D;
```

| 常量 | 值 | 含义 |
|---|---|---|
| `TRACK_D` | 150 px | 一直在追的槽位，门限 |
| `RECOVER_D` | 280 px | 丢过的槽位，门限 |
| `TOLERATE` | 45 帧 | 容忍帧数 |
| `D_R_MIN / D_R_MAX` | 40 / 250 px | 目标到 R 标的环带范围 |


---

## 四、已知失败情况

**Task3**：最开始r识别不到，视频快结束时候，一直将科技核心识别为r,使得连线一直在改变，通过设置异常帧短路解决id跳动

**Task4**：画面跳动的时候，目标圆位置改变，判定为了新的圆，导致id在视频中期快速改变和增加，有时候r标没有识别到，也导致距离判定使得id增加


---

## 五、结果视频

每个任务各输出两个视频，都在 `result/task3_windmill/` 下：

### Task3（`result/task3_windmill/task3/`）

- [recognition_overlay.mp4](task3_windmill/task3/recognition_overlay.mp4) — 识别叠加：绿圈 + ID、R 标蓝十字、目标→R标红线、TRACK / LOST 状态
- [binary_process.mp4](task3_windmill/task3/binary_process.mp4) — 二值化过程：用于核对颜色分割与形态学效果

### Task4（`result/task3_windmill/task4/`）

- [recognition_overlay.mp4](task3_windmill/task4/recognition_overlay.mp4) — 识别叠加：两个目标的绿圈 + ID、左上角 HUD 逐槽位显示 `TRACKING` / `LOSS (n/45)`
- [binary_process.mp4](task3_windmill/task4/binary_process.mp4) — 二值化过程
