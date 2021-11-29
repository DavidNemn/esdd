#pragma once

#ifndef MOTION_CORRECTION_HPP
#define MOTION_CORRECTION_HPP

#include <dvs_msgs/Event.h>
#include <ros/time.h>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <vector>

namespace motion_correction
{
    using EventArray = std::vector<dvs_msgs::Event>;

    // 取整
    static inline int int_floor(float x)
    {
        int i = (int)x;
        return i - (i > x);
    }

    // 重置矩阵大小
    static inline void resetMat(cv::Mat &arr, const cv::Size &size, int type = CV_32F)
    {
        if (arr.cols == 0 || arr.rows == 0)
            arr = cv::Mat::zeros(size, type);
        else
            arr.setTo(0);
    }

    // 运动补偿的参数
    class WarpUpdateParams
    {
    public:
        int warp_mode;             // 仅支持cv::MOTION_HOMOGRAPHY单应变换
        int num_pyramid_levels;    // 用于估计warp的图像金字塔个数
        cv::Size sensor_size;      // CV格式的图像大小
        cv::TermCriteria criteria; // 优化终止参数

        WarpUpdateParams() {}

        WarpUpdateParams(int nIt, double eps, int mode, int lvls, cv::Size sensor_size)
            : warp_mode(mode),
              num_pyramid_levels(lvls),
              criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, nIt, eps), /* nIt迭代次数要求, eps精度要求 */
              sensor_size(sensor_size)
        {
        }
    };
    // 初始化warp
    void initWarp(cv::Mat &warp, const WarpUpdateParams &params);
    // 更新warp
    void updateWarp(cv::Mat &warp, const cv::Mat &img0, const cv::Mat &img1, const WarpUpdateParams &params);
    // 计算光流
    cv::Mat computeFlowFromWarp(const cv::Mat &warp, double dt,
                                cv::Size sensor_size,
                                std::vector<cv::Point2f> rectified_points);
    // 把ev_first~ev_last期间的事件累积到sensor_size大小的out图像中
    void drawEventsUndistorted(EventArray::iterator ev_first, EventArray::iterator ev_last,
                               cv::Mat &out,
                               cv::Size sensor_size, const std::vector<cv::Point2f> &rectified_points,
                               const bool use_polarity);
    // 把ev_first~ev_last期间的光流累积到sensor_size大小的out图像中
    void drawEventsMotionCorrectedOpticalFlow(EventArray::iterator ev_first, EventArray::iterator ev_last,
                                              const cv::Mat &flow_field, cv::Mat &out,
                                              cv::Size sensor_size, const std::vector<cv::Point2f> &rectified_points,
                                              const bool use_polarity);

} // namespace motion_correction

#endif