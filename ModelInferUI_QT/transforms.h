#ifndef TRANSFORMS_H
#define TRANSFORMS_H
#include <opencv2/opencv.hpp>

inline void rescale(cv::Mat src, cv::Mat& dst, int target_width, int target_height, float& scale_factor)
{
    int src_w = src.cols;
    int src_h = src.rows;
    if(src_w == target_width && src_h == target_width) {
        dst = src.clone();
        scale_factor = 1.0;
        return;
    }
    // 计算实际比例因子
    scale_factor = std::min(target_width * 1.0 / src_w, target_height * 1.0 / src_h);
    cv::Mat _src;
    int new_w = (int)(src_w * scale_factor + 0.5);
    int new_h = (int)(src_h * scale_factor + 0.5);
    cv::resize(src, _src, cv::Size(new_w, new_h));
    // 移动roi图像
    if (src.channels() == 1) dst = cv::Mat::zeros(target_height, target_width, CV_8UC1);
    else dst = cv::Mat::zeros(target_height, target_width, CV_8UC3);
    cv::Rect roi = cv::Rect(0, 0, new_w, new_h);
    cv::Mat _dst = dst(roi);
    _src.copyTo(_dst);
    return;
}

#endif // TRANSFORMS_H
