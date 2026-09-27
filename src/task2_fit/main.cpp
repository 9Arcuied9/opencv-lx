#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>            //Ceres Solver: 非线性最小二乘
#include <Eigen/Dense>              //3x3 线性最小二乘(粗扫初值用)
#include <limits>
#include <string>
using namespace cv;
using namespace std;
double omega_calculate(double deltat, double theta_x, double theta_xplus2)
{
    double diff = theta_xplus2 - theta_x;
    while (diff > acos(-1.0))
        diff -= 2 * acos(-1.0);
    while (diff < -acos(-1.0))
        diff += 2 * acos(-1.0);
    double omega = diff / deltat;
    return omega;
}



//=====绘图=======

static const int PLOT_L = 70, PLOT_R = 20, PLOT_T = 40, PLOT_B = 50;   // 四周留白(像素)

// 坐标映射图像像素坐标

inline Point toPix(double x, double y, const Size& sz,
                   double xmin, double xmax, double ymin, double ymax)
{
    //缩放因子
    double sx = (sz.width  - PLOT_L - PLOT_R) / (xmax - xmin);
    double sy = (sz.height - PLOT_T - PLOT_B) / (ymax - ymin);
   
    return Point(PLOT_L + cvRound((x - xmin) * sx),
                 sz.height - PLOT_B - cvRound((y - ymin) * sy));
}

// 画坐标框 + 刻度线 + 刻度数字
void drawAxes(Mat& canvas, double xmin, double xmax, double ymin, double ymax)
{
    Size sz = canvas.size();
    //
    Point bl = toPix(xmin, ymin, sz, xmin, xmax, ymin, ymax);
    Point tr = toPix(xmax, ymax, sz, xmin, xmax, ymin, ymax);
    rectangle(canvas, Rect(bl.x, tr.y, tr.x - bl.x, bl.y - tr.y), Scalar(0, 0, 0), 1, LINE_8);
    // 2) 把 x/y 轴各等分 5 段, 在每个端点画刻度并写数字
    for (int i = 0; i <= 5; ++i) {
        double fx = xmin + (xmax - xmin) * i / 5;  
        double fy = ymin + (ymax - ymin) * i / 5;   
        Point px = toPix(fx, ymin, sz, xmin, xmax, ymin, ymax);   // 
        Point py = toPix(xmin, fy, sz, xmin, xmax, ymin, ymax);   //
        line(canvas, px, px + Point(0, 6), Scalar(0, 0, 0), 1, LINE_8);   // 
        line(canvas, py, py + Point(-6, 0), Scalar(0, 0, 0), 1, LINE_8);  // 
        ostringstream osx, osy;
        osx << fixed << setprecision(2) << fx;     
        osy << fixed << setprecision(2) << fy;
        putText(canvas, osx.str(), px + Point(-18, 24), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_8);  // 写 x 数字
        putText(canvas, osy.str(), py + Point(-68, 6), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_8);   // 写 y 数字
    }
}

// 画折线
void drawCurve(const vector<double>& x, const vector<double>& y, Mat& canvas, Scalar color,
               double xmin, double xmax, double ymin, double ymax, int thickness = 2)
{
    Size sz = canvas.size();
    
    for (size_t i = 1; i < x.size(); ++i)
        line(canvas, toPix(x[i - 1], y[i - 1], sz, xmin, xmax, ymin, ymax),
                   toPix(x[i],     y[i],     sz, xmin, xmax, ymin, ymax),
             color, thickness, LINE_AA);
}




struct SineResidual {
    SineResidual(double t, double omega) : x_(t), y_(omega) {}
    // 参数块 p = [b, A, Om, phi], 与示例中 ab[2] 同风格: 一个数组装全部待估参数
    template <typename T>
    bool operator()(const T* const p, T* residual) const {
        
        residual[0] = T(y_) - (p[0] + p[1] * ceres::sin(p[2] * T(x_) + p[3]));
        return true;
    }

    double x_, y_;
};

int main(){
    VideoCapture cap("resources/task_2.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    VideoWriter writer("result/task2_fit/output_result.mp4", 
                        VideoWriter::fourcc('m','p','4','v'),
                        fps,
                        Size(width, height));

    Mat frame; 
    vector<double> t_list;
    vector<double> theta_list;
    vector<double> omega_list;
    int frame_count = 0;
    while (true)
    {
        
        bool ret = cap.read(frame);
        if (!ret) {
            break; 
        }
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        Mat hsv,channels[3], colorEdges , target;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(80, 120, 100), Scalar(150, 255, 255), target);
        morphologyEx(target, target, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5, 5)));
        Canny(target, colorEdges, 100, 200); 
        findContours(target, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
       
        
        double area = contourArea(contours[0]);       
        Rect boundingBox = boundingRect(contours[0]);
        rectangle(frame, boundingBox, Scalar(0, 0, 255), 0.5);
        Point2f targetcenter ,center;
        center = Point2f(480,360);
        float radius;
        
        minEnclosingCircle(contours[0], targetcenter, radius);
        double theta = atan2(center.y - targetcenter.y, targetcenter.x - center.x);
        double omega = 0.0;
        if (frame_count != 0) {
            omega = omega_calculate(1/ fps, theta_list[frame_count-1], theta);
        }
        t_list.push_back(frame_count / fps);
        theta_list.push_back(theta);
        omega_list.push_back(omega);
        ostringstream oss1,oss2,oss3;
        oss1 << "theta=" << fixed << setprecision(2) << theta;
        string text1 = oss1.str();
        oss2 << "t=" << fixed << setprecision(2) << frame_count / fps;
        string text2 = oss2.str();
        oss3 << "omega=" << fixed << setprecision(2) << omega;
        string text3 = oss3.str();
        putText(frame, text1+text2+text3, Point(targetcenter.x, targetcenter.y+50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        writer.write(frame);
        frame_count++;
    }
    
    //用来拟合的样本点
    const int kSample = 7;
    vector<double> t_fit, theta_fit, omega_fit;
    for (size_t i = kSample; i < theta_list.size(); i += kSample) {
        double omega = omega_calculate(kSample / fps, theta_list[i - kSample], theta_list[i]);
        t_fit.push_back(t_list[i]);
        theta_fit.push_back(theta_list[i]);
        omega_fit.push_back(omega);
    }

    // ---------- 用 Ceres 拟合 ω(t) = b + A·sin(Ωt + φ) ----------
    double p[4] = {1.3, 0.5, 1.6, 0.5}; // 初值
    ceres::Problem problem;
    for (size_t i = 0; i < t_fit.size(); ++i){
        auto* cost =
        new ceres::AutoDiffCostFunction<SineResidual, 1, 4>(
            new SineResidual(t_fit[i], omega_fit[i]));

            problem.AddResidualBlock(cost, nullptr, p);
    }
   
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    std::cout << "Estimated parameters: b=" << p[0] << ", A=" << p[1] << ", Ω=" << p[2] << ", φ=" << p[3] << "\n";

    
    vector<double> t_raw(t_list.begin() + 1, t_list.end());            // 舍弃第 0 帧
    vector<double> omega_raw(omega_list.begin() + 1, omega_list.end());

    vector<double> t_curve, omega_curve;                             
    const int kCurvePts = 600;
    for (int i = 0; i < kCurvePts; ++i) {
        double t = t_raw.front() + (t_raw.back() - t_raw.front()) * i / (kCurvePts - 1);
        t_curve.push_back(t);
        omega_curve.push_back(p[0] + p[1] * sin(p[2] * t + p[3]));
    }

    // 1) 统一量程
    double xmin = *min_element(t_raw.begin(), t_raw.end());
    double xmax = *max_element(t_raw.begin(), t_raw.end());
    double ymin = *min_element(omega_raw.begin(), omega_raw.end());
    double ymax = *max_element(omega_raw.begin(), omega_raw.end());
    double dx = (xmax - xmin) * 0.05, dy = (ymax - ymin) * 0.05;
    xmin -= dx; xmax += dx; ymin -= dy; ymax += dy;

   
    Mat fit_comparison(1000, 1500, CV_8UC3, Scalar(255, 255, 255));
    Mat residuals = fit_comparison.clone();

    //坐标框 + 刻度
    drawAxes(fit_comparison, xmin, xmax, ymin, ymax);

    Mat angular_velocity = fit_comparison.clone();

 //第一张图   ==========================================
    //两条曲线
    drawCurve(t_raw, omega_raw, fit_comparison, Scalar(185, 185, 185), xmin, xmax, ymin, ymax, 1);  // 灰: 原始点
    drawCurve(t_curve, omega_curve, fit_comparison, Scalar(0, 0, 200), xmin, xmax, ymin, ymax, 2);  // 红: Ceres 拟合曲线

   
    putText(fit_comparison, "fit_comparison", Point(PLOT_L, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2, LINE_8);

    // 图例
    const char* legend_names[2] = {"original sample", "Ceres fit"};
    Scalar legend_colors[2] = {Scalar(128, 128, 128), Scalar(0, 0, 200)};
    for (int i = 0; i < 2; ++i) {
        int ly = 55 + i * 26;
        line(fit_comparison, Point(1130, ly), Point(1170, ly), legend_colors[i], 3, LINE_8);
        putText(fit_comparison, legend_names[i], Point(1180, ly + 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_8);
    }

//第二张图
    drawCurve(t_curve, omega_curve, angular_velocity, Scalar(0, 0, 200), xmin, xmax, ymin, ymax, 2);
    putText(angular_velocity, "angular_velocity", Point(PLOT_L, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2, LINE_8);
            int ly = 55 ;
            line(angular_velocity, Point(1130, ly), Point(1170, ly), Scalar(0, 0, 200), 3, LINE_8);
            putText(angular_velocity, "angular_velocity", Point(1180, ly + 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_8);

//第三张图
    vector<double> r_list;  
    for (size_t i = 0; i < t_fit.size(); ++i)
        r_list.push_back(omega_fit[i] - (p[0] + p[1] * sin(p[2] * t_fit[i] + p[3])));

    // 残差 y 轴以 0 为中心对称
    double r_abs = 0.0;
    for (double r : r_list) r_abs = max(r_abs, fabs(r));
    double rmin = -1.2 * r_abs, rmax = 1.2 * r_abs;

    drawAxes(residuals, xmin, xmax, rmin, rmax);            
    line(residuals, toPix(xmin, 0.0, residuals.size(), xmin, xmax, rmin, rmax),
                    toPix(xmax, 0.0, residuals.size(), xmin, xmax, rmin, rmax),
         Scalar(0, 0, 0), 1, LINE_8);                       // 零线
    drawCurve(t_fit, r_list, residuals, Scalar(200, 0, 0), xmin, xmax, rmin, rmax, 1);

    putText(residuals, "residual", Point(PLOT_L, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2, LINE_8);
    line(residuals, Point(1130, 55), Point(1170, 55), Scalar(200, 0, 0), 3, LINE_8);
    putText(residuals, "residual", Point(1180, 60),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_8);

    // 打印 RMSE
    double rmse = 0.0;
    for (double r : r_list) rmse += r * r;
    rmse = sqrt(rmse / r_list.size());
    std::cout << fixed << setprecision(6) << "RMSE = " << rmse << "\n";

    //保存
    imwrite("result/task2_fit/fit_comparison.png", fit_comparison);
    imwrite("result/task2_fit/angular_velocity.png", angular_velocity);
    imwrite("result/task2_fit/residuals.png", residuals);

    writer.release();
    cap.release();
    return 0;
}