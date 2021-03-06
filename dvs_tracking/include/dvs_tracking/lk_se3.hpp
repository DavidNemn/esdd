#ifndef LK_SE3_H
#define LK_SE3_H

#include <dvs_msgs/Event.h>
#include <image_geometry/pinhole_camera_model.h>
#include <pcl/common/transforms.h>
#include <Eigen/StdVector>
#include <deque>
#include <vector>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <ros/ros.h>

// 光流跟踪基类
class LKSE3
{
    typedef Eigen::Matrix<float, 6, 6> Matrix6;
    typedef Eigen::Matrix<float, 6, 1> Vector6;

public:
    typedef pcl::PointXYZ Point;
    typedef pcl::PointCloud<Point> PointCloud;
    typedef std::deque<dvs_msgs::Event> EventQueue;

    typedef Sophus::SE3f SE3;

protected:
    size_t max_iterations_; // 最大迭代次数, 5
    size_t pyramid_levels_; // 图像金字塔层数

    int map_blur_; // 把点云投影到关键帧时的模糊程度

    std::vector<Eigen::Vector3f> keypoints; // 关键点坐标
    std::vector<float> pixel_values;        // 像素值
    std::vector<Vector6> J;                 // 雅克比
    std::vector<Matrix6> JJt;               // J*J^T

    PointCloud::Ptr map_;       // 基于关键帧构建的地图
    PointCloud::Ptr map_local_; // 在当前帧可视的局部地图

    cv::Mat depth_kf_;              // 关键帧的深度图
    cv::Mat kf_img_;                // 关键帧
    cv::Mat new_img_;               // 当前帧
    std::vector<cv::Mat> pyr_new_;  // 当前帧new_img_的图像金字塔
    cv::Mat grad_x_img, grad_y_img; // xy方向上的梯度图像

    float *grad_x, *grad_y;

    image_geometry::PinholeCameraModel c_;    // 当前帧相机模型
    image_geometry::PinholeCameraModel c_kf_; // 关键帧相机模型
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    size_t width_;
    size_t height_;
    cv::Rect rect_; // 图像大小, width×height

    Eigen::Isometry3f T_;          // 当前帧->世界坐标系
    Eigen::Isometry3f T_curr_;     // 关键帧->当前帧
    Eigen::Isometry3f T_curr_inv_; // 当前帧->关键帧
    Eigen::Isometry3f T_kf_;       // 关键帧->世界坐标系
    Eigen::Isometry3f T_wb_;       // 机体系->世界坐标系
    Eigen::Isometry3f T_bc_;       // 相机系->机体系
    Eigen::Isometry3f T_bc_inv_;   // 机体系->相机系

    Eigen::VectorXf x_ = Vector6::Zero(); // T_curr_的李代数表示

    void projectMap();                         // 投影3D点云到关键帧
    void precomputereferenceFrame();           // 计算关键帧的关键点
    void computecurrentFrame();                // 计算当前帧的雅克比
    void trackFrame();                         // tracking函数
    void updateTransformation(size_t pyr_lvl); // tracking子函数

    void drawEvents(EventQueue::iterator ev_first, EventQueue::iterator ev_last, cv::Mat &out); // 把ev_first到ev_last的事件累积到out

    Eigen::Vector3f b_w_;    // 陀螺仪偏置
    Eigen::Vector3f b_a_;    // 加速度计偏置
    Eigen::Vector3f g_;      // 重力
    Eigen::Vector3f v_last_; // 上一时刻速度(机体系下)
    Eigen::Vector3f v_;      // 速度(机体系下)
};
#endif