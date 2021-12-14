#include "dvs_tracking/tracker.hpp"

#include <camera_info_manager/camera_info_manager.h>
#include <cv_bridge/cv_bridge.h>
#include <dvs_msgs/EventArray.h>
#include <eigen_conversions/eigen_msg.h>
#include <image_transport/image_transport.h>
#include <kindr/minimal/quat-transformation.h>
#include <nav_msgs/Path.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/Marker.h>

#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <thread>

#include "evo_utils/camera.hpp"
#include "evo_utils/math.hpp"
#include "evo_utils/main.hpp"
#include "rpg_common_ros/params_helper.hpp"

using Transformation = kindr::minimal::QuatTransformation; // 旋转矩阵
using Quaternion = kindr::minimal::RotationQuaternion;     // 四元数

Tracker::Tracker(ros::NodeHandle &nh, ros::NodeHandle nh_private)
    : nh_(nh), nhp_(nh_private), it_(nh_), tf_(true, ros::Duration(2.)), cur_ev_(0), kf_ev_(0),
      noise_rate_(nhp_.param("noise_rate", 10000)), frame_size_(nhp_.param("frame_size", 2500)),
      step_size_(nhp_.param("step_size", 2500)), idle_(true)
{
    max_iterations_ = nhp_.param("max_iterations", 100);
    map_blur_ = nhp_.param("map_blur", 5);
    pyramid_levels_ = nhp_.param("pyramid_levels", 1);

    T_ = T_curr_inv_ = T_kf_ = T_wb_ = Eigen::Isometry3f::Identity();
    b_w_ << 0, 0, 0;
    b_a_ << 0, 0, 0;
    g_ << 0, 0, -9.8;
    v_last_.setZero();
    v_.setZero();
    Eigen::Matrix4f T_bc_tmp;
    T_bc_tmp << 9.9987240400523691e-01, -1.4310300709482299e-02, 7.0986620142410852e-03, 6.8356317644019523e-03,
        1.4285328717366487e-02, 9.9989163566394357e-01, 3.5561654517748472e-03, -8.1401134374014996e-03,
        -7.1487825694325933e-03, -3.4543049789795689e-03, 9.9996848084571499e-01, 4.0618002505661763e-02,
        0, 0, 0, 1.000000000000000;
    T_bc_ = T_bc_tmp;
    T_bc_inv_ = T_bc_.inverse();

    map_ = PointCloud::Ptr(new PointCloud);
    map_local_ = PointCloud::Ptr(new PointCloud);

    c_ = evo_utils::camera::loadPinholeCamera(nh);
    postCameraLoaded();

    event_sub_ = nh_.subscribe("events", 0, &Tracker::eventCallback, this);
    map_sub_ = nh_.subscribe("pointcloud", 0, &Tracker::mapCallback, this);
    tf_sub_ = nh_.subscribe("tf", 0, &Tracker::tfCallback, this);
    poses_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("evo/pose", 0);

    imu_sub_.subscribe(nh_, "imu_topic", 500, ros::TransportHints().tcpNoDelay());
    imu_cache_.reset(new message_filters::Cache<sensor_msgs::Imu>(imu_sub_, 100));
    imu_cache_->registerCallback(boost::bind(&Tracker::imuCb, this, _1));
#ifdef TRACKER_DEBUG_REFERENCE_IMAGE
    std::thread map_overlap(&Tracker::publishMapOverlapThread, this);
    map_overlap.detach();
#endif

    frame_id_ = rpg_common_ros::param(nh_, "dvs_frame_id", std::string("dvs_evo"));
    world_frame_id_ = rpg_common_ros::param(nh_, "world_frame_id", std::string("world"));
    auto_trigger_ = rpg_common_ros::param<bool>(nhp_, "auto_trigger", true);

    std::thread tracker(&Tracker::trackingThread, this);
    tracker.detach();
}

void Tracker::postCameraLoaded()
{
    // 加载相机参数
    width_ = c_.fullResolution().width;
    height_ = c_.fullResolution().height;
    fx_ = c_.fx();
    fy_ = c_.fy();
    cx_ = c_.cx();
    cy_ = c_.cy();
    rect_ = cv::Rect(0, 0, width_, height_);

    float fov = 2. * std::atan(c_.fullResolution().width / 2. / c_.fx()); // 计算当前帧视场角fov
    LOG(INFO) << "Field of view: " << fov / M_PI * 180.;

    new_img_ = cv::Mat(c_.fullResolution(), CV_32F, cv::Scalar(0)); // 设置当前帧图像大小

    sensor_msgs::CameraInfo cam_last = c_.cameraInfo(); // 关键帧相机坐标
    cam_last.width = nh_.param("virtual_width", c_.fullResolution().width);
    cam_last.height = nh_.param("virtual_height", c_.fullResolution().height);
    cam_last.P[0 * 4 + 2] = cam_last.K[0 * 3 + 2] = 0.5 * (float)cam_last.width;
    cam_last.P[1 * 4 + 2] = cam_last.K[1 * 3 + 2] = 0.5 * (float)cam_last.height;

    float f_last = nh_.param("fov_virtual_camera_deg", 0.); // 关键帧视场角fov_last
    if (f_last == 0.)
        f_last = c_.fx();
    else
    {
        const float f_last_rad = f_last * CV_PI / 180.0;
        f_last = 0.5 * (float)cam_last.width / std::tan(0.5 * f_last_rad);
    }
    cam_last.P[0 * 4 + 0] = cam_last.K[0 * 3 + 0] = f_last;
    cam_last.P[1 * 4 + 1] = cam_last.K[1 * 3 + 1] = f_last;
    c_kf_.fromCameraInfo(cam_last);

    reset();
}

void Tracker::reset()
{
    idle_ = true;            // 设置线程状态
    events_.clear();         // 清空事件
    poses_.clear();          // 清空位姿
    poses_filtered_.clear(); // 清空滤波位姿
    cur_ev_ = kf_ev_ = 0;    // 事件编号清零
}

void Tracker::eventCallback(const dvs_msgs::EventArray::ConstPtr &msg)
{
    // 仅添加事件到events_
    static const bool discard_events_when_idle = rpg_common_ros::param<bool>(nhp_, "discard_events_when_idle", false); // 空闲时丢弃到来的事件

    std::lock_guard<std::mutex> lock(data_mutex_);
    if (discard_events_when_idle && idle_)
        return;

    clearEventQueue();
    for (const auto &e : msg->events)
        events_.push_back(e);
}

void Tracker::mapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    static size_t min_map_size = rpg_common_ros::param<int>(nhp_, "min_map_size", 0); // map的最小容量

    std::lock_guard<std::mutex> lock(data_mutex_);
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(*msg, pcl_pc);
    pcl::fromPCLPointCloud2(pcl_pc, *map_);

    if (map_->size() > min_map_size && auto_trigger_)
    {
        LOG(INFO) << "Auto-triggering tracking";

        initialize(ros::Time(0)); // 初始化
        auto_trigger_ = false;
    }
}

void Tracker::imuCb(const sensor_msgs::Imu::ConstPtr &msg)
{
}

std::tuple<Eigen::Matrix3f, Eigen::Vector3f, Eigen::Vector3f, float, Eigen::Matrix<float, 9, 9>>
Tracker::imuIntegrate(std::vector<sensor_msgs::Imu::ConstPtr> &imu_vector) const
{
    // 一段时间里的imu观测
    int num_measurement = imu_vector.size();
    Eigen::Matrix3f R_meas;
    Eigen::Vector3f v_meas;
    Eigen::Vector3f p_meas;
    float t_meas = 0;
    R_meas << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    v_meas << 0, 0, 0;
    p_meas << 0, 0, 0;
    // 协方差
    Eigen::Matrix<float, 9, 9> Cov_meas;
    if (imu_vector.size() != 0)
        Cov_meas.setZero();
    else
        Cov_meas.setIdentity();
    // 误差传递矩阵
    Eigen::Matrix<float, 9, 9> A;
    A.setIdentity();
    // 噪声传递矩阵
    Eigen::Matrix<float, 9, 6> B;
    B.setZero();
    // 获取imu测量
    for (int i = 0; i < num_measurement - 1; i++)
    {
        // 获取陀螺仪和加速度计测量值
        Eigen::Vector3f w;
        Eigen::Vector3f a;
        w(0) = imu_vector[i]->angular_velocity.x;
        w(1) = imu_vector[i]->angular_velocity.y;
        w(2) = imu_vector[i]->angular_velocity.z;
        a(0) = imu_vector[i]->linear_acceleration.x;
        a(1) = imu_vector[i]->linear_acceleration.y;
        a(2) = imu_vector[i]->linear_acceleration.z;
        // 去除偏差(预设值,不参与更新)
        w = w - b_w_;
        a = a - b_a_;
        // 获取δt
        float delta_t = (imu_vector[i + 1]->header.stamp - imu_vector[i]->header.stamp).toSec();
        // 更新传递矩阵
        Eigen::Matrix<float, 3, 3> delta_R_jm1_j = evo_utils::math::exp_SO3<float>(w * delta_t);
        Eigen::Matrix<float, 3, 3> a_hat = evo_utils::math::hat(a);
        A.block<3, 3>(0, 0) = delta_R_jm1_j.transpose();
        A.block<3, 3>(3, 0) = -R_meas * a_hat * delta_t;
        A.block<3, 3>(6, 0) = -0.5 * R_meas * a_hat * delta_t * delta_t;
        A.block<3, 3>(6, 3) = delta_t * Eigen::MatrixXf::Identity(3, 3);
        B.block<3, 3>(0, 0) = evo_utils::math::exp_Jacobian<float>(w * delta_t) * delta_t;
        B.block<3, 3>(3, 3) = R_meas * delta_t;
        B.block<3, 3>(6, 3) = 0.5 * R_meas * delta_t * delta_t;
        // 更新协方差矩阵
        Cov_meas = A * Cov_meas * A.transpose() + B * Cov_noise_ * B.transpose();
        // 更新状态量
        t_meas += delta_t;
        R_meas *= delta_R_jm1_j;
        v_meas += R_meas * a * delta_t;
        p_meas += v_meas * delta_t + 0.5 * R_meas * a * delta_t * delta_t;
    }
    // 返回C++元组
    return std::make_tuple(R_meas, v_meas, p_meas, t_meas, Cov_meas);
}

void Tracker::initialize(const ros::Time &ts)
{
    // 仅在第一次接收到map的时候使用，用于得到最先的位姿，发布后在mapping部分切换
    std::string bootstrap_frame_id = rpg_common_ros::param<std::string>(nh_, "dvs_bootstrap_frame_id", std::string("/camera0")); // 初始化id
    // 坐标变换, world->初始化帧
    tf::StampedTransform TF_kf_world;
    Eigen::Isometry3d T_kf_world;
    tf_.lookupTransform(bootstrap_frame_id, world_frame_id_, ts, TF_kf_world);
    tf::transformTFToEigen(TF_kf_world, T_kf_world);

    T_ = T_kf_world.cast<float>().inverse();
    T_kf_ = T_;
    T_curr_inv_ = Eigen::Isometry3f::Identity();
    T_wb_ = T_bc_ * T_ * T_bc_inv_;

    while (cur_ev_ + 1 < events_.size() && events_[cur_ev_].ts < TF_kf_world.stamp_)
        ++cur_ev_;

    updateMap();
    idle_ = false;
}

void Tracker::updateMap()
{
    static size_t min_map_size = rpg_common_ros::param<int>(nhp_, "min_map_size", 0);       // map的最小容量
    static size_t min_n_keypoints = rpg_common_ros::param<int>(nhp_, "min_n_keypoints", 0); // 关键点的最小个数

    if (map_->size() <= min_map_size)
    {
        LOG(WARNING) << "Unreliable map! Can not update map."; // 没有点云, 报错!
        return;
    }

    T_kf_ = T_kf_ * T_curr_inv_;
    T_curr_inv_ = Eigen::Isometry3f::Identity();
    kf_ev_ = cur_ev_;

    projectMap(); // 投影点云得到关键帧的关键点和雅克比(FCA)

    if (keypoints.size() < min_n_keypoints)
    {
        LOG(WARNING) << "Losing track!"; // 关键点数量不够, tracking失败!
    }
}

void Tracker::trackingThread()
{
    // 以100Hz的频率进行tracking
    static ros::Rate r(100);
    while (ros::ok())
    {
        r.sleep();
        if (!idle_ && keypoints.size() > 0)
        {
            estimateTrajectory();
        }
    }
}

void Tracker::estimateTrajectory()
{
    static const size_t max_event_rate = nhp_.param("max_event_rate", 8000000); // 最大事件频率, 如果事件频率超过噪声频率就不累积
    static const size_t events_per_ref = nhp_.param("events_per_kf", 100000);   // 每events_per_kf个事件构建一个关键帧

    std::lock_guard<std::mutex> lock(data_mutex_);

    while (true)
    {
        if (cur_ev_ + std::max(step_size_, frame_size_) > events_.size())
            break;

        // 超过events_per_ref就构建关键帧
        if (cur_ev_ - kf_ev_ >= events_per_ref)
            updateMap();

        if (idle_)
            break;

        size_t frame_end = cur_ev_ + frame_size_;

        double frameduration = (events_[frame_end].ts - events_[cur_ev_].ts).toSec(); // 累积事件的时间戳范围
        event_rate_ = std::round(static_cast<double>(frame_size_) / frameduration);   // 累积事件频率
        if (event_rate_ < noise_rate_)
        {
            LOG(WARNING) << "Event rate below NOISE RATE. Skipping frame.";
            cur_ev_ += step_size_;
            continue;
        }
        if (event_rate_ > max_event_rate)
        {
            LOG(WARNING) << "Event rate above MAX EVENT RATE. Skipping frame.";
            cur_ev_ += step_size_;
            continue;
        }

        std::vector<sensor_msgs::Imu::ConstPtr> imu_vector = imu_cache_->getInterval(events_[cur_ev_].ts, events_[frame_end].ts);
        Eigen::Matrix3f R_meas;
        Eigen::Vector3f v_meas;
        Eigen::Vector3f p_meas;
        float t_meas;
        Eigen::Matrix<float, 9, 9> Cov_meas;
        std::tie(R_meas, v_meas, p_meas, t_meas, Cov_meas) = imuIntegrate(imu_vector);
        // 导入两帧之间imu的更新量
        importImuMeas(R_meas, v_meas, p_meas, t_meas, Cov_meas);

        drawEvents(events_.begin() + cur_ev_, events_.begin() + frame_end, new_img_); // 把frame_size_个事件累积到new_img_
        cv::buildPyramid(new_img_, pyr_new_, pyramid_levels_);                        // 构建图像金字塔
        trackFrame();                                                                 // 进行tracking

        publishTF();           // 发布tf
        cur_ev_ += step_size_; // 跳过step_size_个事件
    }
}

void Tracker::importImuMeas(const Eigen::Matrix3f &R_meas, const Eigen::Vector3f &v_meas, const Eigen::Vector3f &p_meas, const float t_meas, const Eigen::Matrix<float, 9, 9> &Cov_meas)
{
    R_meas_ = R_meas;
    v_meas_ = v_meas;
    p_meas_ = p_meas;
    t_meas_ = t_meas;
    Cov_meas_ = Cov_meas;
}

void Tracker::tfCallback(const tf::tfMessagePtr &msgs)
{
    // 用于空闲时获取初始化位姿
    if (!idle_)
        return;

    for (auto &msg : msgs->transforms)
    {
        tf::StampedTransform t;
        tf::transformStampedMsgToTF(msg, t);
        tf_.setTransform(t);
    }
}

void Tracker::clearEventQueue()
{
    static size_t event_history_size_ = 500000;

    if (idle_)
    {
        if (events_.size() > event_history_size_)
        {
            events_.erase(events_.begin(), events_.begin() + events_.size() - event_history_size_);
            cur_ev_ = kf_ev_ = 0;
        }
    }
    else // 创建了kf就清理掉
    {
        events_.erase(events_.begin(), events_.begin() + kf_ev_);
        cur_ev_ -= kf_ev_;
        kf_ev_ = 0;
    }
}

void Tracker::publishMapOverlapThread()
{
    static ros::Rate r(nhp_.param("event_map_overlap_rate", 25)); // 以r的频率发布overlap图像
    static image_transport::Publisher pub = it_.advertise("event_map_overlap", 1);
    cv::Mat ev_img, img;

    while (ros::ok())
    {
        r.sleep();

        if (idle_ || pub.getNumSubscribers() == 0 || event_rate_ < noise_rate_)
            continue;

        cv::convertScaleAbs(1. - .25 * new_img_, ev_img, 255);
        cv::cvtColor(ev_img, img, cv::COLOR_GRAY2RGB);

        std_msgs::Header header;
        header.stamp = events_[cur_ev_].ts;

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
        pub.publish(msg);
    }
}

void Tracker::publishTF()
{
    // 发布T_world_cur
    tf::Transform pose_tf;
    tf::transformEigenToTF(T_.cast<double>(), pose_tf);
    // 注意时间戳, 累积了frame_size_的事件图像时间戳就是事件的时间戳
    tf::StampedTransform new_pose(pose_tf, events_[cur_ev_ + frame_size_].ts, world_frame_id_, "dvs_evo_raw");
    poses_.push_back(new_pose);
    // tf_pub_.sendTransform(new_pose);

    tf::StampedTransform filtered_pose;
    if (getFilteredPose(filtered_pose))
    {
        filtered_pose.frame_id_ = world_frame_id_;
        filtered_pose.child_frame_id_ = frame_id_;
        tf_pub_.sendTransform(filtered_pose);
        poses_filtered_.push_back(filtered_pose);

        publishPose();
    }
}

bool Tracker::getFilteredPose(tf::StampedTransform &pose)
{
    static const size_t mean_filter_size = nhp_.param("pose_mean_filter_size", 10);

    if (mean_filter_size < 2)
    {
        pose = poses_.back();
        return true;
    }

    if (poses_.size() < mean_filter_size)
        return false;

    static Eigen::VectorXd P(7);
    P.setZero();

    tf::Quaternion tf_q0 = poses_[poses_.size() - mean_filter_size].getRotation();
    const Quaternion q0(tf_q0.w(), tf_q0.x(), tf_q0.y(), tf_q0.z());
    const Quaternion q0_inv = q0.inverse();

    for (size_t i = poses_.size() - mean_filter_size; i != poses_.size(); ++i)
    {
        const tf::Quaternion &tf_q = poses_[i].getRotation();
        const Quaternion q(tf_q.w(), tf_q.x(), tf_q.y(), tf_q.z());
        const Quaternion q_inc = q0_inv * q;

        const tf::Vector3 &t = poses_[i].getOrigin();

        Transformation T(q_inc, Eigen::Vector3d(t.x(), t.y(), t.z()));

        P.head<6>() += T.log();
        P[6] += poses_[i].stamp_.toSec();
    }

    P /= mean_filter_size;
    Transformation T(Transformation::Vector6(P.head<6>()));

    const Eigen::Vector3d &t_mean = T.getPosition();
    const Quaternion q_mean = q0 * T.getRotation();

    tf::StampedTransform filtered_pose;
    filtered_pose.setOrigin(tf::Vector3(t_mean[0], t_mean[1], t_mean[2]));
    filtered_pose.setRotation(tf::Quaternion(q_mean.x(), q_mean.y(), q_mean.z(), q_mean.w()));
    filtered_pose.stamp_ = ros::Time(P[6]);

    pose = filtered_pose;
    return true;
}

void Tracker::publishPose()
{
    const tf::StampedTransform &T_world_cam = poses_.back();

    const tf::Vector3 &p = T_world_cam.getOrigin();
    const tf::Quaternion &q = T_world_cam.getRotation();
    geometry_msgs::PoseStampedPtr msg_pose(new geometry_msgs::PoseStamped);
    msg_pose->header.stamp = T_world_cam.stamp_;
    msg_pose->header.frame_id = frame_id_;
    msg_pose->pose.position.x = p.x();
    msg_pose->pose.position.y = p.y();
    msg_pose->pose.position.z = p.z();
    msg_pose->pose.orientation.x = q.x();
    msg_pose->pose.orientation.y = q.y();
    msg_pose->pose.orientation.z = q.z();
    msg_pose->pose.orientation.w = q.w();
    poses_pub_.publish(msg_pose);
}

// void Tracker::publishImg()
// {
//     cv::Mat ev_img;
//     cv::convertScaleAbs(1. - .25 * new_img_, ev_img, 255);
//     static cv_bridge::CvImage cv_image;
//     cv_image.encoding = "mono8";
//     cv_image.header.stamp = ros::Time::now();
//     cv_image.image = new_img_.clone();
//     new_image_pub_.publish(cv_image.toImageMsg());
// }
