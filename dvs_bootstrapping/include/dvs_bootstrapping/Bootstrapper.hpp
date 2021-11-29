#pragma once

#ifndef BOOTSTRAPPING_HPP
#define BOOTSTRAPPING_HPP

#include <dvs_msgs/EventArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <image_geometry/pinhole_camera_model.h>
#include <ros/node_handle.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include "evo_utils/camera.hpp"
#include <mutex>
// 系统初始化基类
namespace dvs_bootstrapping
{
    class Bootstrapper
    {
    public:
        Bootstrapper(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
        virtual ~Bootstrapper() {}

        void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg); // 相机参数

    protected:
        virtual void postCameraLoaded() {} // 相机参数赋值

        ros::NodeHandle nh_, nhp_;               // ros句柄
        image_geometry::PinholeCameraModel cam_; // 相机模型

        bool idle_ = true; // 正在初始化

        std::vector<dvs_msgs::Event> eventQueue_; // 保存事件
        std::mutex data_mutex_;                   // 处理事件时加锁

        std::string world_frame_id_;     // 系统的world坐标系
        std::string bootstrap_frame_id_; // 初始化时的坐标系

    private:
        ros::Subscriber camera_info_sub_; // 用于cameraInfoCalback
        ros::Subscriber event_sub_;       // 用于eventCallback

        void eventCallback(const dvs_msgs::EventArray::ConstPtr &msg); // 事件流累积
    };

} // namespace dvs_bootstrapping

#endif