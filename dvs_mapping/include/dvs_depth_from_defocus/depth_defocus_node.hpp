#pragma once

#include <geometry_msgs/PoseStamped.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <tf/tf.h>
#include <tf/tfMessage.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include <boost/multi_array.hpp>
#include <dvs_depth_from_defocus/depth_vector.hpp>

#include "dvs_msgs/Event.h"
#include "dvs_msgs/EventArray.h"
#include "evo_utils/camera.hpp"
#include "evo_utils/geometry.hpp"

// #define MAPPING_PERF // 测试mapping线程的性能

namespace depth_from_defocus
{
    typedef pcl::PointXYZI PointType;
    typedef pcl::PointCloud<PointType> PointCloud;
    typedef float Vote;

    enum MapperState
    {
        IDLE,
        MAPPING
    };

    class DepthFromDefocusNode
    {
    public:
        DepthFromDefocusNode(const ros::NodeHandle &nh, const ros::NodeHandle &pnh, const image_geometry::PinholeCameraModel &cam);

        ~DepthFromDefocusNode();

        void processEventArray(const dvs_msgs::EventArray::ConstPtr &event_array); // 事件回调函数
        void tfCallback(const tf::tfMessage::ConstPtr &tf_msg);                    // 坐标变换回调函数
        void remoteKeyCallback(const std_msgs::String::ConstPtr &msg);             // 地图更新回调函数
        // void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg);     // 相机参数回调函数

    private:
        void resetMapper();    // 初始化mapping线程
        void setupVoxelGrid(); // 体素划分与深度切片

        void update();                              // mapping核心操作
        void resetVoxelGrid();                      // 网格置零, 获取关键帧位姿
        void processEventQueue(bool reset = false); // 获取事件位姿
        void projectEventsToVoxelGrid(const std::vector<Eigen::Vector4f> &events,
                                      const std::vector<Eigen::Vector3f> &centers); // 投影并更新体素
        bool getPoseAt(const ros::Time &t, evo_utils::geometry::Transformation &T); // 根据线性插值获取某时间戳处的位姿

        void voteForCell(const float x_f, const float y_f, Vote *grid)
        {
            const int x = (int)(x_f + 0.5), y = (int)(y_f + 0.5);
            if (x >= 0 && x < virtual_width_ && y >= 0 && y < virtual_height_)
            {
                grid[x + y * virtual_width_] += 1.f;
            }
        }

        void voteForCellBilinear(const float x_f, const float y_f, Vote *grid)
        {
            if (x_f >= 0.f && y_f >= 0.f)
            {
                const int x = x_f, y = y_f;
                if (x + 1 < virtual_width_ && y + 1 < virtual_height_)
                {
                    Vote *g = grid + x + y * virtual_width_;
                    const float fx = x_f - x, fy = y_f - y, fx1 = 1.f - fx, fy1 = 1.f - fy;

                    g[0] += fx1 * fy1;
                    g[1] += fx * fy1;
                    g[virtual_width_] += fx1 * fy;
                    g[virtual_width_ + 1] += fx * fy;
                }
            }
        }

        void precomputeRectifiedPoints(); // 去畸变

        void synthesizeAndPublishDepth();                                                                 // 计算深度并发布
        void synthesizePointCloudFromVoxelGrid(cv::Mat &depth, cv::Mat &confidence);                      // 从体素中获取深度
        void synthesizePointCloudFromVoxelGridLinf(cv::Mat &depth_cell_indices, cv::Mat &confidence);     // Linf范数
        void synthesizePointCloudFromVoxelGridContrast(cv::Mat &depth_cell_indices, cv::Mat &confidence); // 最大对比度
        void synthesizePointCloudFromVoxelGridGradMag(cv::Mat &depth_cell_indices, cv::Mat &confidence);  // 最大梯度幅度

        void accumulatePointcloud(const cv::Mat &depth, const cv::Mat &mask,
                                  const cv::Mat &confidence,
                                  const Eigen::Matrix3d R_world_ref,
                                  const Eigen::Vector3d t_world_ref,
                                  const ros::Time &timestamp);                           // 通过深度合成点云并发布
        void publishVoxelGrid(const evo_utils::geometry::Transformation &T_w_ref) const; // 发布体素网格
        void publishDepthmap(const cv::Mat &depth, const cv::Mat &mask);                 // 发布深度图
        void publishGlobalMap();                                                         // 发布全局地图

        void clearEventQueue();  // 清空事件队列
        void postCameraLoaded(); // 获取相机参数

        inline const Vote &voxelGridAt(const std::vector<Vote> &grid, int x, int y, size_t z) const
        {
            return grid[x + virtual_width_ * (y + z * virtual_height_)];
        }

        inline Vote &voxelGridAt(std::vector<Vote> &grid, int x, int y, size_t z)
        {
            return grid[x + virtual_width_ * (y + z * virtual_height_)];
        }

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;
        image_transport::ImageTransport it_;

        ros::Publisher pub_pc_;                    // 发布局部点云
        ros::Publisher pub_pc_global_;             // 发布全局点云
        ros::Publisher pub_voxel_grid_;            // 发布体素网格
        image_transport::Publisher pub_depth_map_; // 发布参考帧处的深度图

        ros::Subscriber event_sub_;  // 事件
        ros::Subscriber tf_sub_;     // 坐标变换
        ros::Subscriber remote_key_; // 地图更新

        MapperState state_; // 建图线程状态

        std::string world_frame_id_;          // 世界坐标系id
        std::string frame_id_;                // 相机坐标系id
        std::string regular_frame_id_;        // 常规相机坐标系id
        std::string bootstrap_frame_id_;      // 初始化相机坐标系id
        std::shared_ptr<tf::Transformer> tf_; // 坐标变换

        evo_utils::geometry::Transformation T_ref_w_; // 相机位姿

        image_geometry::PinholeCameraModel dvs_cam_; // 相机模型
        int width_;
        int height_;

        evo_utils::camera::PinholeCamera virtual_cam_; // 扩展针孔相机模型
        int virtual_width_;
        int virtual_height_;

        Eigen::Matrix3f K_; // 相机内参

        std::vector<Vote> ref_voxel_grid_; // 体素投票结果
        cv::Mat confidence_mask_;          // 置信度
        float voxel_filter_leaf_size_;     // 滤波规模
        size_t events_to_recreate_kf_;     // 新建关键帧的最大事件数目
        int type_focus_measure_;           // 计算DSI极大值时的方式, 默认使用Linf范数, 值为0

        Eigen::Matrix2Xf precomputed_rectified_points_;

        size_t current_event_;                    // 当前处理的事件
        size_t newest_tracked_event_;             // 位姿未知的事件
        size_t last_kf_update_event_;             // 更新地图的最后一个事件
        std::deque<dvs_msgs::Event> event_queue_; // 存储事件

        PointCloud::Ptr pc_;        // 局部地图
        PointCloud::Ptr pc_global_; // 全局地图

        evo_utils::geometry::Depth min_depth_; // 最小深度
        evo_utils::geometry::Depth max_depth_; // 最大深度
        size_t num_depth_cells_;               // 深度切片数目
        InverseDepthVector depths_vec_;        // 存储DSI
        std::vector<float> raw_depths_vec_;    // 存储像素深度

        float radius_search_;    // 半径滤波
        int min_num_neighbors_;  // 用于剔除外点
        int median_filter_size_; // 中值滤波

        int adaptive_threshold_kernel_size_; // 滤波核
        int adaptive_threshold_c_;           // 滤波阈值

        bool auto_trigger_; // 是否接收到第一个位姿就开始mapping
    };
} // namespace depth_from_defocus
