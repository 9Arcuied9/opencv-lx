#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// ===================================================================================
// 能量机关双目标追踪
//
// 两个槽位, 每个槽位记录：
//     pos        目标圆的位置
//     id         目标圆的 id
//     tracking   是否追踪到(本帧有没有认到自己的圆)
//     lostFrames 容忍帧数(连续没认到多少帧了)
//
// 每帧处理：
//   步骤0  颜色掩膜 + 轮廓 + 分类  -> 本帧场上所有目标圆
//   步骤1  R 标定位(带短期记忆)
//   步骤2  几何粗筛(目标应落在 R 标周围的环带上)
//   步骤3  匹配：
//     场上只有一个圆 -> 先算它到两个槽位的距离是否都过长：
//          都过长            -> 不是相同的圆 -> 给它新 id, 存进其中一个槽位
//          不是都过长        -> 同一个圆 -> 离得更近的槽位继承(id 不变, 位置更新),
//                               另一个槽位记为 loss(其他保持不变)
//     场上不止一个圆 -> 每个槽位取离自己最近的那个圆
//   步骤4  没认到自己的圆的槽位 -> 容忍帧数 +1, 过了容忍帧数才把记录清空
//   步骤5  剩下的圆 = 新圆 -> 存进空槽位(或挤掉丢得最久的槽位), 给新 id
//   步骤6  绘制 + HUD
// ===================================================================================

struct Slot {
    Point2f pos;          // 目标圆的位置
    float   radius;       // 目标圆半径
    int     id;           // 目标圆的 id;
    bool    tracking;     // 是否追踪到
    int     lostFrames;   // 容忍帧数
};

int main() {
    VideoCapture cap2("resources/task_4.mp4");
    if (!cap2.isOpened()) { 
        cerr << "Error opening task_4.mp4" << endl; 
        return -1; 
    }

    int w2 = (int)cap2.get(CAP_PROP_FRAME_WIDTH);
    int h2 = (int)cap2.get(CAP_PROP_FRAME_HEIGHT);
    double fps2 = cap2.get(CAP_PROP_FPS);
    VideoWriter writer2("result/task3_windmill/task4/recognition_overlay.mp4",
                        VideoWriter::fourcc('m', 'p', '4', 'v'), fps2, Size(w2, h2));

    // 两个槽位
    Slot slots[2] = {
        {Point2f(0, 0), 0.f, -1, false, 0},
        {Point2f(0, 0), 0.f, -1, false, 0}
    };
    int nextId = 1;         

    // ---------------- 参数 ----------------
    const float TRACK_D   = 150.0f;  
                                                                     
    const float RECOVER_D = 280.0f;  
    const int   TOLERATE  = 45;      
    const float D_R_MIN   = 40.0f;   
    const float D_R_MAX   = 250.0f;  

  
    Point2f lastR(0, 0);                  
    int     lostRFrames = 999;            
    Point2f imgCenter(w2 / 2.0f, h2 / 2.0f); 

    Mat frame;
    while (cap2.read(frame)) {
        //  提取橙黄色区域
       
        Mat hsv, mask1, mask2, mask;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(0, 40, 50),  Scalar(10, 255, 255), mask1);
        inRange(hsv, Scalar(170, 40, 50), Scalar(179, 255, 255), mask2);
        bitwise_or(mask1, mask2, mask);
        dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));

        //需要层级信息区分靶心
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        Mat overlay = frame.clone();
        struct Circle { Point2f c; float r; int nChild; double area; };
        vector<Circle> circles;

        // 轮廓筛选

        for (size_t i = 0; i < contours.size(); ++i) {
            if (hierarchy[i][3] != -1) continue;
            double area = contourArea(contours[i]);
            if (area < 300.0) continue;
            double peri = arcLength(contours[i], true);
            if (peri <= 0.0) continue;
            double circ = 4.0 * CV_PI * area / (peri * peri);   // 圆度=4πA/P², 正圆=1
            Point2f center; float radius;
            minEnclosingCircle(contours[i], center, radius);
            if (circ >= 0.6) {
                // nChild = 直接子轮廓数: 靶心(同心环+辐条)子轮廓多, 普通小圆子轮廓少
                int nChild = 0;
                for (int c = hierarchy[i][2]; c != -1; c = hierarchy[c][0]) ++nChild;
                circles.push_back({center, radius, nChild, area});
            }
        }

       
        vector<Point2f> curPts; vector<float> curRs;   // 目标靶心
        vector<Circle> rCands;                        
        for (const Circle& c : circles) {
            if (c.area >= 4000.0) {
                if (c.nChild >= 2) { curPts.push_back(c.c); curRs.push_back(c.r); }
            } else if (c.area >= 400.0 && c.area <= 1100.0) {
                rCands.push_back(c);
            }
        }

      //即使没识别到r,仍可存储之前的r的位置
        Point2f rC(0, 0); bool haveR = false;
        if (!rCands.empty()) {
            Point2f refPt = imgCenter;
            if (!curPts.empty()) {                   
                Point2f cen(0, 0);
                for (const Point2f& p : curPts) cen += p;
                refPt = cen * (1.0f / curPts.size());
            } else if (lostRFrames < 30) {         
                refPt = lastR;
            }

            float bestD = 1e9f; int bestB = -1;
            for (size_t k = 0; k < rCands.size(); ++k) {
                float d = (float)norm(rCands[k].c - refPt);
                if (d < bestD) { bestD = d; bestB = (int)k; }
            }
            if (bestB >= 0 && bestD <= 100.0f) { rC = rCands[bestB].c; haveR = true; }
        }

        if (haveR) {
            lastR = rC; lostRFrames = 0;
        } else if (lostRFrames < 30) {
            rC = lastR; haveR = true; lostRFrames++;
        }

       
        if (haveR) {
            vector<Point2f> fp; vector<float> fr;
            for (size_t i = 0; i < curPts.size(); ++i) {
                float dist = (float)norm(curPts[i] - rC);
                if (dist >= D_R_MIN && dist <= D_R_MAX) {
                    fp.push_back(curPts[i]);
                    fr.push_back(curRs[i]);
                }
            }
            curPts = fp; curRs = fr;
        }

        int k = (int)curPts.size();
        vector<bool> used(k, false);
        bool gotIt[2] = {false, false};         // 本帧每个槽位有没有认到自己的圆

    
        int   pickJ[2] = {-1, -1};             
        float pickD[2] = {1e9f, 1e9f};          
        float lim[2];                           
        for (int i = 0; i < 2; ++i) lim[i] = slots[i].tracking ? TRACK_D : RECOVER_D;

        if (k == 1) {
            for (int i = 0; i < 2; ++i) {              
                if (slots[i].id == -1) continue;        
                pickD[i] = (float)norm(curPts[0] - slots[i].pos);
                pickJ[i] = 0;
            }
            if (pickD[0] > lim[0] && pickD[1] > lim[1]) {
                pickJ[0] = pickJ[1] = -1;               // ② 都过长 -> 不是相同的圆
            } else {
                int near = (pickD[0] <= pickD[1]) ? 0 : 1;   // ③ 同一个圆 -> 更近的槽位继承
                pickJ[1 - near] = -1;                        //    另一个槽位记为 loss
            }
        } else {
            for (int i = 0; i < 2; ++i) {               // 每个槽位取离自己最近的圆
                if (slots[i].id == -1) continue;
                for (int j = 0; j < k; ++j) {
                    float d = (float)norm(curPts[j] - slots[i].pos);
                    if (d <= lim[i] && d < pickD[i]) { pickD[i] = d; pickJ[i] = j; }
                }
            }
            if (pickJ[0] >= 0 && pickJ[0] == pickJ[1]) {          
                int near = (pickD[0] <= pickD[1]) ? 0 : 1;
                int far  = 1 - near;
                pickJ[far] = -1; pickD[far] = 1e9f;
                for (int j = 0; j < k; ++j) {                     
                    if (j == pickJ[near]) continue;
                    float d = (float)norm(curPts[j] - slots[far].pos);
                    if (d <= lim[far] && d < pickD[far]) { pickD[far] = d; pickJ[far] = j; }
                }
            }
        }

        // 继承: id 不变, 只更新位置/半径
        for (int i = 0; i < 2; ++i) {
            if (pickJ[i] < 0) continue;
            slots[i].pos        = curPts[pickJ[i]];
            slots[i].radius     = curRs[pickJ[i]];
            slots[i].tracking   = true;
            slots[i].lostFrames = 0;
            gotIt[i]            = true;
            used[pickJ[i]]      = true;
        }

        
        
        
        for (int i = 0; i < 2; ++i) {
            if (slots[i].id == -1) continue;        
            if (gotIt[i]) continue;                
            slots[i].tracking = false;
            if (++slots[i].lostFrames > TOLERATE) {
                slots[i].id = -1;
                slots[i].lostFrames = 0;
            }
        }

       
   
        for (int j = 0; j < k; ++j) {
            if (used[j]) continue;
            int slot = -1;
            for (int i = 0; i < 2; ++i) if (slots[i].id == -1) { slot = i; break; }
            if (slot < 0) {
                int worst = 0;
                for (int i = 0; i < 2; ++i) {
                    if (slots[i].lostFrames > worst) { worst = slots[i].lostFrames; slot = i; }
                }
            }
            if (slot < 0) continue;                 
            slots[slot].pos        = curPts[j];
            slots[slot].radius     = curRs[j];
            slots[slot].id         = nextId++;      // id 更新
            slots[slot].tracking   = true;
            slots[slot].lostFrames = 0;
            used[j]                = true;
        }

    
        if (haveR) {                                  // R 标: 蓝点 + 蓝十字
            line(overlay, rC - Point2f(16, 0), rC + Point2f(16, 0), Scalar(255, 0, 0), 2);
            line(overlay, rC - Point2f(0, 16), rC + Point2f(0, 16), Scalar(255, 0, 0), 2);
            circle(overlay, rC, 5, Scalar(255, 0, 0), -1);
        }
        for (int i = 0; i < 2; ++i) {                 // 追踪到的: 绿圈 + ID
            if (!slots[i].tracking) continue;
            if (haveR) line(overlay, slots[i].pos, rC, Scalar(0, 100, 255), 2);
            circle(overlay, slots[i].pos, (int)slots[i].radius, Scalar(0, 255, 0), 2);
            putText(overlay, format("ID:%d", slots[i].id),
                    slots[i].pos + Point2f(slots[i].radius, -slots[i].radius),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        }

        // 左上角 HUD: 逐个槽位显示状态
        rectangle(overlay, Rect(10, 10, 260, 80), Scalar(0, 0, 0, 160), -1);
        int hudY = 40;
        bool anyActive = false;
        for (int i = 0; i < 2; ++i) {
            if (slots[i].id == -1) continue;
            anyActive = true;
            if (slots[i].tracking) {
                putText(overlay, format("ID:%d | TRACKING", slots[i].id), Point(20, hudY),
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            } else {
                putText(overlay, format("ID:%d | LOSS (%d/%d)", slots[i].id, slots[i].lostFrames, TOLERATE),
                        Point(20, hudY), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
            }
            hudY += 30;
        }
        if (!anyActive) {
            putText(overlay, "ID: -- | NO TARGET", Point(20, 40),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }

        writer2.write(overlay);
    }

    writer2.release();
    cap2.release();

    //二值化过程
    {
        VideoCapture capB("resources/task_4.mp4");
        if (!capB.isOpened()) {
            cerr << "Error opening task_4.mp4" << endl;
            return -1;
        }

        VideoWriter bwWriter("result/task3_windmill/task4/binary_process.mp4",
                             VideoWriter::fourcc('m', 'p', '4', 'v'),
                             fps2,
                             Size(w2, h2));
        if (!bwWriter.isOpened()) {
            cerr << "Error opening binary_process.mp4 (目录是否存在?)" << endl;
            return -1;
        }

        Mat fB;
        while (capB.read(fB)) {
            Mat hsvB, n1, n2, mB;
            cvtColor(fB, hsvB, COLOR_BGR2HSV);
            inRange(hsvB, Scalar(0, 40, 50),   Scalar(10, 255, 255), n1);
            inRange(hsvB, Scalar(170, 40, 50), Scalar(179, 255, 255), n2);
            bitwise_or(n1, n2, mB);

            dilate(mB, mB, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));

            Mat canvas;
            cvtColor(mB, canvas, COLOR_GRAY2BGR);   // 只写膨胀后的二值图
            bwWriter.write(canvas);
        }
        bwWriter.release();
        capB.release();
    }
    return 0;
}