#ifndef TRACKER_H
#define TRACKER_H

#include <dvs_msgs/EventArray.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>

#include <sensor_msgs/Imu.h>
#include <message_filters/cache.h>
#include <message_filters/subscriber.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <mutex>

#include "dvs_tracking/lk_se3.hpp"

#define TRACKER_EVENT_HISTORY_SIZE 500000 // 保留TRACKER_EVENT_HISTORY_SIZE个历史事件
#define TRACKER_DEBUG_REFERENCE_IMAGE     // 开启图像覆盖线程

class Tracker : public LKSE3
{
public:
    Tracker(ros::NodeHandle &nh, ros::NodeHandle nh_private);

private:
    ros::NodeHandle nh_, nhp_;           // ros句柄
    image_transport::ImageTransport it_; // 图像
    tf::Transformer tf_;                 // tf
    tf::TransformBroadcaster tf_pub_;    // 发布坐标变换

    ros::Subscriber event_sub_;  // 事件
    ros::Subscriber map_sub_;    // 地图
    ros::Subscriber remote_sub_; // 远程消息
    ros::Subscriber tf_sub_;     // 坐标变换
    ros::Publisher poses_pub_;   // 位姿

    std::string frame_id_;       // 当前坐标id
    std::string world_frame_id_; // 世界坐标系id

    bool idle_;         // tracking线程是否在运行, false为在运行
    bool auto_trigger_; // 默认自动运行

    EventQueue events_; // 处理事件

    size_t cur_ev_;     // 当前事件编号
    size_t kf_ev_;      // 关键帧事件编号
    size_t noise_rate_; // 噪声频率, 如果事件频率低于噪声频率就不累积
    size_t frame_size_; // 累积的事件个数, frame_size_前后为两帧连续的图像
    size_t step_size_;  // 下次累积事件的起始步长
    size_t event_rate_; // 累积事件频率

    std::vector<tf::StampedTransform> poses_;          // 估计的位姿
    std::vector<tf::StampedTransform> poses_filtered_; // 中值滤波后的位姿

    std::mutex data_mutex_; // 处理事件加锁

    void eventCallback(const dvs_msgs::EventArray::ConstPtr &msg);   // 事件回调函数
    void mapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg); // 地图回调函数
    void tfCallback(const tf::tfMessagePtr &msgs);                   // 坐标变换回调函数

    void initialize(const ros::Time &ts);             // 初始化
    void reset();                                     // 重置/清空工作
    void trackingThread();                            // tracking线程
    void estimateTrajectory();                        // 实际tracking
    bool getFilteredPose(tf::StampedTransform &pose); // 位姿滤波
    void updateMap();                                 // 更新关键帧处的地图

    void publishMapOverlapThread(); // 发布地图覆盖图像
    void publishTF();               // 发布坐标变换
    void publishPose();             // 发布当前位姿

    //--------------------------------------------------------------------------------------
    Eigen::Matrix<float, 6, 6> Cov_noise_;
    void imuCb(const sensor_msgs::Imu::ConstPtr &msg);
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    std::unique_ptr<message_filters::Cache<sensor_msgs::Imu>> imu_cache_; // imu缓存, 直接把msg导入!
    std::tuple<Eigen::Matrix3f, Eigen::Vector3f, Eigen::Vector3f, float, Eigen::Matrix<float, 9, 9>>
    imuIntegrate(std::vector<sensor_msgs::Imu::ConstPtr> &imu_vector) const;
    //--------------------------------------------------------------------------------------

    void clearEventQueue();  // 清空事件队列
    void postCameraLoaded(); // 加载相机参数
};

#endif // TRACKER_H
