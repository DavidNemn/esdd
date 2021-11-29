#pragma once

#ifndef KLT_BOOTSTRAPPER_HPP
#define KLT_BOOTSTRAPPER_HPP

#include <image_transport/image_transport.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <deque>
#include <opencv2/core.hpp>
#include <vector>

#include "dvs_bootstrapping/Bootstrapper.hpp"
#include "motion_correction.hpp"

// 系统具体的初始化类
namespace dvs_bootstrapping
{
    class EventsFramesBootstrapper : public Bootstrapper
    {
    public:
        EventsFramesBootstrapper(ros::NodeHandle &nh, ros::NodeHandle &nhp);
        virtual ~EventsFramesBootstrapper();

    protected:
        void postCameraLoaded() override; // 相机参数赋值(重写)

        size_t width_;                              // 图像宽
        size_t height_;                             // 图像高
        cv::Size sensor_size_;                      // 图像CV形式大小
        std::vector<cv::Point2f> rectified_points_; // 校正后像素位置
        std::deque<cv::Mat> frames_;                // 累积的事件图像
        int rate_hz_;                               // 累积事件的频率
        ros::Time ts_;                              // 最后一张累积图像的时间戳

    private:
        tf::Transformer tf_;
        tf::TransformBroadcaster tf_pub_;
        image_transport::ImageTransport it_;
        image_transport::Publisher pub_event_img_;

        int frame_size_;             // 窗口中的事件数量
        int local_frame_size_;       // 实际累积到图像中的事件数量
        int newest_processed_event_; // 最近处理的事件计数
        bool enable_visuals_;        // 发布事件累积图像
        cv::Mat undistort_mapx_;     // x方向去畸变
        cv::Mat undistort_mapy_;     // y方向去畸变

        void integratingThread(); // 累积线程
        bool integrateEvents();   // 累积事件处理函数
        void clearEventQueue();   // 清空队列

        void publishEventImage(const cv::Mat &img, const ros::Time &ts); // 发布事件图像
    };

} // namespace dvs_bootstrapping

#endif