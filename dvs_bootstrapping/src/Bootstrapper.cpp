#include "dvs_bootstrapping/Bootstrapper.hpp"

#include <camera_info_manager/camera_info_manager.h>
#include <geometry_msgs/PoseStamped.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sensor_msgs/CameraInfo.h>
#include <string.h>

#include "rpg_common_ros/params_helper.hpp"

namespace dvs_bootstrapping
{

    Bootstrapper::Bootstrapper(ros::NodeHandle &nh, ros::NodeHandle &nhp)
        : nh_(nh), nhp_(nhp)
    {
        idle_ = !rpg_common_ros::param<bool>(nhp_, "auto_trigger", true); // 一般为false
        LOG(INFO) << "Bootstrapper initially idle: " << idle_;

        cam_ = evo_utils::camera::loadPinholeCamera(nh_); // 加载相机模型

        // 订阅/发布
        event_sub_ = nh_.subscribe("events", 0, &Bootstrapper::eventCallback, this);
        camera_info_sub_ = nh_.subscribe("camera_info", 1, &Bootstrapper::cameraInfoCallback, this);

        // 坐标系
        world_frame_id_ = rpg_common_ros::param<std::string>(nh_, "world_frame_id", std::string("world"));
        bootstrap_frame_id_ = rpg_common_ros::param(nh_, "dvs_bootstrap_frame_id", std::string("/dvs_evo"));
    }

    void Bootstrapper::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg)
    {
        static bool got_camera_info = false;

        if (!got_camera_info)
        {
            cam_.fromCameraInfo(*msg);

            postCameraLoaded();

            // Currently, done only once
            got_camera_info = true;
        }
    }

    void Bootstrapper::eventCallback(const dvs_msgs::EventArray::ConstPtr &msg)
    {
        if (idle_)
            return;

        std::lock_guard<std::mutex> lock(data_mutex_);
        eventQueue_.insert(eventQueue_.end(), msg->events.begin(), msg->events.end());
    }

} // namespace dvs_bootstrapping