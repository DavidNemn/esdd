#include "dvs_bootstrapping/EventsFramesBootstrapper.hpp"

#include <cv_bridge/cv_bridge.h>
#include <std_msgs/String.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

#include "evo_utils/geometry.hpp"
#include "rpg_common_ros/params_helper.hpp"

namespace dvs_bootstrapping
{
    EventsFramesBootstrapper::~EventsFramesBootstrapper() {}
    EventsFramesBootstrapper::EventsFramesBootstrapper(ros::NodeHandle &nh, ros::NodeHandle &nhp)
        : Bootstrapper(nh, nhp), it_(nh)
    {
        postCameraLoaded();

        frame_size_ = rpg_common_ros::param<int>(nhp, "frame_size", 50000);             // 窗口中的事件数量
        local_frame_size_ = rpg_common_ros::param<int>(nhp, "local_frame_size", 10000); // 实际累积到图像中的事件数量
        CHECK_GE(frame_size_, local_frame_size_);                                       // 窗口中的事件数量 > 实际累积到图像中的事件数量

        rate_hz_ = rpg_common_ros::param<int>(nhp, "rate_hz", 25);
        enable_visuals_ = rpg_common_ros::param<bool>(nhp, "enable_visualizations", true);
        if (enable_visuals_)
            pub_event_img_ = it_.advertise(rpg_common_ros::param<std::string>(nhp, "motion_corrected_topic", "/evo/bootstrap/event_frame"), 1);

        std::thread integrate(&EventsFramesBootstrapper::integratingThread, this);
        integrate.detach();
    }

    void EventsFramesBootstrapper::postCameraLoaded()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        sensor_size_ = cam_.fullResolution();
        height_ = sensor_size_.height;
        width_ = sensor_size_.width;

        cv::initUndistortRectifyMap(cam_.intrinsicMatrix(), cam_.distortionCoeffs(),
                                    cv::noArray(), cv::noArray(), sensor_size_,
                                    CV_32FC1, undistort_mapx_, undistort_mapy_);

        eventQueue_.clear();

        evo_utils::camera::precomputeRectificationTable(rectified_points_, cam_);
    }

    void EventsFramesBootstrapper::integratingThread()
    {
        // 固定r的频率发布事件图像
        static ros::Rate r(rate_hz_);

        while (ros::ok())
        {
            r.sleep();

            if (!idle_)
                if (integrateEvents())
                    // 累积成功就清空事件队列
                    clearEventQueue();
        }
    }

    void EventsFramesBootstrapper::clearEventQueue()
    {
        if (newest_processed_event_ <= frame_size_)
            return;

        std::lock_guard<std::mutex> lock(data_mutex_);
        // 留下最近的frame_size_个事件
        eventQueue_.erase(eventQueue_.begin(), eventQueue_.begin() + (newest_processed_event_ - frame_size_));
    }

    bool EventsFramesBootstrapper::integrateEvents()
    {
        static cv::Mat img1, event_img;
        static std::vector<dvs_msgs::Event> events;

        static size_t min_step_size = rpg_common_ros::param<int>(nhp_, "min_step_size", 5000);           // 最少要处理min_step_size个事件
        static float events_scale_factor = rpg_common_ros::param<float>(nhp_, "events_scale_factor", 7); // 像素的放大倍数

        std::lock_guard<std::mutex> lock_events(data_mutex_);
        if (eventQueue_.size() <= frame_size_)
            return false;

        size_t last_event = eventQueue_.size() - 1;
        if (last_event - newest_processed_event_ < min_step_size)
            return false;

        auto frame1_begin = eventQueue_.end() - local_frame_size_;
        auto frame1_end = eventQueue_.end() - 1;

        motion_correction::resetMat(img1, sensor_size_, CV_32F);
        motion_correction::resetMat(event_img, sensor_size_, CV_32F);

        motion_correction::drawEventsUndistorted(frame1_begin, frame1_end, img1, sensor_size_, rectified_points_, false);

        cv::Mat frame = (255.0 / events_scale_factor) * (img1); // 事件图像
        frame.convertTo(frame, CV_8U);                          // 归一化数值:0~255
        ts_ = frame1_end->ts;                                   // 时间戳
        if (enable_visuals_)
            publishEventImage(frame, ts_);

        newest_processed_event_ = last_event;
        return true;
    }

    void EventsFramesBootstrapper::publishEventImage(const cv::Mat &img, const ros::Time &ts)
    {
        static cv_bridge::CvImage cv_event_image;

        cv_event_image.encoding = "mono8";
        if (pub_event_img_.getNumSubscribers() > 0)
        {
            img.convertTo(cv_event_image.image, CV_8U);

            auto aux = cv_event_image.toImageMsg();
            aux->header.stamp = ts;
            pub_event_img_.publish(aux);
        }
    }
} // namespace dvs_bootstrapping