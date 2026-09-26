#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main() {
    //检查读图是否成功
    Mat img = imread("../../resources/test_image.jpg");
    if (img.empty()) {
    std::cerr << "Cannot read image!\n";
    return 1;
    }
    
    //输出灰度图以及 H、S、V 单通道图
    Mat gray , hsv , channels[3];
    cvtColor(img, gray, COLOR_BGR2GRAY);
    cvtColor(img, hsv, COLOR_BGR2HSV);
    split(hsv, channels);

    imwrite("../../result/task1_images/gray.png", gray);
    imwrite("../../result/task1_images/hsv_h.png", channels[0]);
    imwrite("../../result/task1_images/hsv_s.png", channels[1]);
    imwrite("../../result/task1_images/hsv_v.png", channels[2]);
    
    //滤波对比
    Mat meanImg, gaussianImg, medianImg;
    blur(img, meanImg, Size(5, 5));
    GaussianBlur(img, gaussianImg, Size(5, 5), 1.5);
    medianBlur(img, medianImg, 5);
    imwrite("../../result/task1_images/mean.png", meanImg);//均值
    imwrite("../../result/task1_images/gaussian.png", gaussianImg);//高斯
    imwrite("../../result/task1_images/median.png", medianImg);//中值

    //红色提取
    Mat mask , maskLow , maskHigh;
    inRange(hsv, Scalar(0,100,80), Scalar(10,255,255), maskLow);
    inRange(hsv, Scalar(170,100,80), Scalar(179,255,255), maskHigh);
    bitwise_or(maskLow, maskHigh, mask);
    imwrite("../../result/task1_images/red_mask.png", mask);
    
    //形态学与轮廓
    Mat erodeImg, dilateImg, openImg, closeImg;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(mask, erodeImg, kernel);
    dilate(mask, dilateImg, kernel);
    morphologyEx(mask, openImg, MORPH_OPEN, kernel);
    morphologyEx(mask, closeImg, MORPH_CLOSE, kernel);
    imwrite("../../result/task1_images/erode.png", erodeImg);
    imwrite("../../result/task1_images/dilate.png", dilateImg);
    imwrite("../../result/task1_images/open.png", openImg);  
    imwrite("../../result/task1_images/close.png", closeImg);

    Mat edges;
    Mat result= img.clone();
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(closeImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(result, contours, -1, Scalar(255, 0, 0), 2);
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 500) continue; 
        Rect box = boundingRect(contours[i]);
        double ratio = static_cast<double>(box.width) / box.height;
        if (ratio < 0.2 || ratio > 5.0) continue;
        rectangle(result, box, Scalar(0, 0, 255), 2);
        drawContours(result, contours, static_cast<int>(i),Scalar(0, 255, 0), 2);
        string text = "Area:" + to_string(area);

        putText(result, text, Point(box.x, box.y - 5),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
    }
    imwrite("../../result/task1_images/contours_boxes.png", result);

    //绘制与转换
    Mat drawing , rotatedImg , cropTopLeftImg;

    // 在原图副本上绘制圆、矩形和文字
    drawing = img.clone();
    circle(drawing, Point(320, 210), 100, Scalar(255, 0, 0), 2);                  //蓝圆
    rectangle(drawing, Point(800, 150), Point(1100, 400), Scalar(0, 255, 0), 2); //绿矩形
    putText(drawing, "Red words", Point(60, 120),FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);//红字
                                 
    imwrite("../../result/task1_images/draw_shapes.png", drawing);

    //绕图像中心旋转 35°
    Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    Mat rot = getRotationMatrix2D(center, 35, 1.0);
    warpAffine(img, rotatedImg, rot, img.size());
    imwrite("../../result/task1_images/rotate35.png", rotatedImg);

    //裁剪原图左上角 1/4
    cropTopLeftImg = img(Rect(0, 0, img.cols / 2, img.rows / 2)).clone();
    imwrite("../../result/task1_images/crop_topleft.png", cropTopLeftImg);
    

}   