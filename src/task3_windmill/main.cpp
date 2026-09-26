#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

int main(){
//task3=======================================================================================
// 

    VideoCapture cap("../../resources/task_3.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;  
    }
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    VideoWriter writer("../../result/task3_windmill/task3/recognition_overlay.mp4", 
                        VideoWriter::fourcc('m', 'p', '4', 'v'),
                        fps,
                        Size(width, height));

    Mat frame; 
    while (true)
    {       
        bool ret = cap.read(frame);
        if (!ret) {
            break; 
        }
        Mat hsv, mask1, mask2, mask;
        
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        
        inRange(hsv, Scalar(0, 40, 50), Scalar(10, 255, 255), mask1);
        inRange(hsv, Scalar(170, 40, 50), Scalar(179, 255, 255), mask2);
        bitwise_or(mask1, mask2, mask);
        
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
        dilate(mask, mask, kernel);     // 合并碎片: 否则中心R标等细结构会被拆碎, 过不了面积/圆度筛选

        //筛选圆型轮廓 (RETR_TREE 拿层级)
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        Mat overlay = frame.clone();

        // 收集轮廓: 圆型(圆度>=0.8) 留作目标圆; 非圆(圆度<0.8)按面积留作找R标
        struct Circle { Point2f c; float r; int nChild; };
        struct Blob   { Point2f c; double area; };
        vector<Circle> circles;
        vector<Blob>   blobs;

        for (size_t i = 0; i < contours.size(); ++i) {
            if (hierarchy[i][3] != -1) continue;        // 只看最外层轮廓

            double area = contourArea(contours[i]);
            if (area < 300.0) continue;                 // 滤掉小噪点/碎片

            double peri = arcLength(contours[i], true);
            if (peri <= 0.0) continue;

            // 圆度 = 4π·面积 / 周长², 正圆为1, 越接近1越像圆
            double circularity = 4.0 * CV_PI * area / (peri * peri);

            Point2f center;
            float   radius;
            minEnclosingCircle(contours[i], center, radius);

            if (circularity >= 0.8) {                   // 圆型轮廓: 目标圆 / 被击中圆环
                // 层级筛选: 目标圆子轮廓>=2(同心环+辐条); 被击中圆环子轮廓=1(内孔)
                int nChild = 0;
                for (int c = hierarchy[i][2]; c != -1; c = hierarchy[c][0]) ++nChild;
                circles.push_back({center, radius, nChild});
            } else if (area <= 1200.0) {                // 非圆: 按面积过滤(去掉臂/黄条等大块) -> R标候选
                blobs.push_back({center, area});
            }
        }

        
        Point2f hub(0, 0);
        int     nHub = 0;
        for (const Circle& c : circles)
            if (c.nChild >= 1) { hub += c.c; ++nHub; }
        if (nHub > 0) hub *= 1.0f / nHub;

        // 目标圆 (子轮廓>=2) -> 只记录目标圆心(不画红圈, 由下面 id追踪 画圈)
        Point2f tCenter(0, 0);
        bool    haveT = false;
        for (const Circle& c : circles)
            if (c.nChild >= 2) {
                tCenter = c.c;
                haveT   = true;
            }

        // R标 = 非圆轮廓中, 离风车中心最近的那个
        int   rIdx = -1;
        float rBest = 1e9f;
        if (nHub > 0)
            for (size_t k = 0; k < blobs.size(); ++k) {
                float d = (float)norm(blobs[k].c - hub);
                if (d < rBest) { rBest = d; rIdx = (int)k; }
            }

        // 连线: 目标圆心 -> R标中心
        if (haveT && rIdx >= 0) {
            Point2f rc = blobs[rIdx].c;
            line(overlay, tCenter, rc, Scalar(0, 0, 255), 2);                // 连线(红)
            circle(overlay, rc, 5, Scalar(255, 0, 0), -1);                   // R标中心(蓝点)
            line(overlay, rc - Point2f(16, 0), rc + Point2f(16, 0), Scalar(255, 0, 0), 2);
            line(overlay, rc - Point2f(0, 16), rc + Point2f(0, 16), Scalar(255, 0, 0), 2);
        }

        // ============ id 追踪: 单目标稳定锁定 ============
    
        static int     tId      = -1;   // 当前目标 ID; -1 = 尚未锁定
        static int     nextId   = 0;    // 下一个待分配的 ID
        static int     lastSeen = 0;    // 最后一次看到目标的帧号
        static int     frameNo  = 0;    // 当前帧号
        static Point2f tPos(0, 0);      // 目标最后已知坐标(丢失期间保持不变)
        static float   tR       = 0.f;  // 目标半径(指数平滑, 减少抖动)

        const float MAX_SAME_D = 150.0f; // 位移门限: 连续追踪时 <=150px 才算同一个圆, 否则换 ID
        const float REAPPEAR_D = 250.0f; // 丢失后重新出现时放宽到 250px(目标消失几帧再回来时位置会飘)
        const int   MAX_LOST   = 30;     // 丢失容忍帧数(30帧≈1秒): 超过才判为换圆
        const float D_R_MAX    = 260.0f; // 到 R 标距离上限: 超过判为异常帧(R 标误检), 此时保持 ID 不变

        // 1. 在候选圆中找离上一帧位置最近的一个
        int bestIdx = -1;
        float minD = 1e9f;
        for (size_t i = 0; i < circles.size(); ++i) {
            if (circles[i].nChild >= 2) {
                float d = (float)norm(circles[i].c - tPos);
                if (d < minD) { minD = d; bestIdx = (int)i; }
            }
        }

        // 2. 目标圆到 R 标距离
        float dR = -1.f;
        if (bestIdx >= 0 && rIdx >= 0)
            dR = (float)norm(circles[bestIdx].c - blobs[rIdx].c);

        // 3. ID 判定 (三步链, 前面成立则后面失效):
        if (bestIdx >= 0) {
            bool farFromR = (rIdx >= 0 && dR > D_R_MAX);   // 第一步: 与 R 标距离过远

            // 第二步位移门限: 连续追踪用 150, 短暂丢失后重新出现放宽到 250
            bool reappear = (tId >= 0 && (frameNo - lastSeen) > 1);
            float sameD   = reappear ? REAPPEAR_D : MAX_SAME_D;

            if (tId < 0) {
                tId = nextId++;                            // 首次锁定
            } else if (farFromR) {
                // 异常帧 -> 同一个圆, ID 不变
            } else if (minD > sameD) {
                tId = nextId++;                            // 第二步: 位移不合适
            } else if ((frameNo - lastSeen) > MAX_LOST) {
                tId = nextId++;                            // 第三步: 丢失超时
            }

            tPos     = circles[bestIdx].c;
            tR       = 0.5f * tR + 0.5f * circles[bestIdx].r; // 平滑半径
            lastSeen = frameNo;

            // 绘制目标绿圈与 ID + 追踪状态
            circle(overlay, tPos, (int)tR, Scalar(0, 255, 0), 2);
            putText(overlay, format("ID:%d", tId), tPos + Point2f(tR, -tR),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            putText(overlay, "TRACK", tPos + Point2f(tR, -tR + 25),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        } else if (tId >= 0) {
            // 检测失败 -> 在最后位置标明丢失
            putText(overlay, format("ID:%d", tId), tPos + Point2f(tR, -tR),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            putText(overlay, "LOST", tPos + Point2f(tR, -tR + 25),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        }

        ++frameNo;

        writer.write(overlay);

    }
    writer.release();
    cap.release();

    //展示二值化过程
    {
        VideoCapture cap2("../../resources/task_3.mp4");
        if (!cap2.isOpened()) {
            std::cerr << "Error opening video stream or file" << std::endl;
            return -1;
        }

        VideoWriter bwWriter("../../result/task3_windmill/task3/binary_process.mp4",
                             VideoWriter::fourcc('m', 'p', '4', 'v'),
                             fps,
                             Size(width, height));
        if (!bwWriter.isOpened()) {
            std::cerr << "Error opening binary_process.mp4 (目录是否存在?)" << std::endl;
            return -1;
        }

        Mat f2;
        while (cap2.read(f2)) {
            Mat hsv2, m1, m2, m;
            cvtColor(f2, hsv2, COLOR_BGR2HSV);
            inRange(hsv2, Scalar(0, 40, 50),   Scalar(10, 255, 255), m1);
            inRange(hsv2, Scalar(170, 40, 50), Scalar(179, 255, 255), m2);
            bitwise_or(m1, m2, m);

            Mat k2 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
            dilate(m, m, k2);

            // 只写膨胀后的二值图
            Mat canvas;
            cvtColor(m, canvas, COLOR_GRAY2BGR);
            bwWriter.write(canvas);
        }
        bwWriter.release();
        cap2.release();
    }
    return 0;
}
