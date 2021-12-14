#include "dvs_tracking/lk_se3.hpp"

#include <glog/logging.h>

#include <boost/range/irange.hpp>
#include <random>

#include "dvs_tracking/weight_functions.hpp"
#include "evo_utils/interpolation.hpp"
#include "evo_utils/math.hpp"

#define DEBUG_PRINT_LIMITS

void LKSE3::projectMap()
{
    static std::vector<float> z_values;
    z_values.clear();
    map_local_->clear();

    cv::Size s = c_kf_.fullResolution();
    cv::Mat &depthmap = depth_kf_;
    depthmap = cv::Mat(s, CV_32F, cv::Scalar(0.));
    cv::Mat img(s, CV_32F, cv::Scalar(0.));

    size_t n_points = 0;
    // 先投影得到kf_img_
    for (const auto &P : map_->points)
    {
        Eigen::Vector3f p(P.x, P.y, P.z);
        p = T_kf_.inverse() * p;
        p[0] = p[0] / p[2] * fx_ + cx_;
        p[1] = p[1] / p[2] * fy_ + cy_;
        z_values.push_back(p[2]);
        ++n_points;
        if (p[0] < 0 || p[1] < 0)
            continue;

        int x = p[0] + .5f, y = p[1] + .5f;
        if (x >= s.width || y >= s.height)
            continue;

        float z = p[2];
        float &depth = depthmap.at<float>(y, x);
        if (depth == 0.)
            depth = z;
        else if (depth > z)
            depth = z;

        img.at<float>(y, x) = 1.;
        map_local_->push_back(P);
    }

    const int k = map_blur_;
    cv::GaussianBlur(img, img, cv::Size(k, k), 0.);
    cv::GaussianBlur(depthmap, depthmap, cv::Size(k, k), 0.);
    depthmap /= img;
    kf_img_ = img;

    precomputereferenceFrame();
}

void LKSE3::precomputereferenceFrame()
{
    // sobel算子计算图像梯度
    cv::Mat grad_x_img, grad_y_img;
    cv::Sobel(kf_img_, grad_x_img, CV_32F, 1, 0);
    cv::Sobel(kf_img_, grad_y_img, CV_32F, 0, 1);
    // 再反投影得到keypoints_
    float *depth_last = depth_kf_.ptr<float>(0);
    grad_x = grad_x_img.ptr<float>(0); // x方向像素梯度
    grad_y = grad_y_img.ptr<float>(0); // y方向像素梯度
    int w = kf_img_.cols, h = kf_img_.rows;

    // 对于图像中梯度值大的点, 加入keypoints_
    keypoints.clear();
    pixel_values.clear();
    J.clear();
    JJt.clear();

    Eigen::Matrix<float, 1, 2> J_grad;
    Eigen::Matrix<float, 2, 3> J_proj;
    Eigen::Matrix<float, 3, 6> J_SE3;

    Vector6 vec = Vector6::Zero();
    Eigen::Map<const Vector6> vec6(&vec(0));

    for (size_t y = 0; y != h; ++y)
    {
        for (size_t x = 0; x != w; ++x)
        {
            size_t offset = y * w + x;
            float z = depth_last[offset]; // 深度
            float pixel_value = kf_img_.at<float>(y, x);

            if (pixel_value < .01)
                continue;

            float X = z * ((float)x - cx_) / fx_;
            float Y = z * ((float)y - cy_) / fy_;
            float Z_inv = 1.f / z, Z2_inv = 1.f / (z * z);

            J_grad(0, 0) = grad_x[offset];
            J_grad(0, 1) = grad_y[offset];

            J_proj(0, 0) = fx_ * Z_inv;
            J_proj(1, 0) = 0;
            J_proj(0, 1) = 0;
            J_proj(1, 1) = fy_ * Z_inv;
            J_proj(0, 2) = -fx_ * X * Z2_inv;
            J_proj(1, 2) = -fy_ * Y * Z2_inv;

            Eigen::Matrix3f npHat;
            npHat << 0, z, -Y, -z, 0, X, Y, -X, 0;
            J_SE3 << Eigen::Matrix3f::Identity(3, 3), npHat;

            vec = -(J_grad * J_proj * J_SE3).transpose();

            keypoints.push_back({X, Y, z});
            pixel_values.push_back(pixel_value);
            J.push_back(vec);
            JJt.push_back(vec6 * vec6.transpose());
        }
    }
}


void LKSE3::updateTransformation(size_t pyr_lvl)
{
    static Eigen::MatrixXf H;
    static Eigen::VectorXf b, dx;
    const cv::Mat &img = pyr_new_[pyr_lvl];
    const float *new_img = img.ptr<float>(0);
    float scale = std::pow(2.f, (float)pyr_lvl);
    float fx = fx_ / scale,
          fy = fy_ / scale,
          cx = cx_ / scale,
          cy = cy_ / scale;
    size_t w = img.cols, h = img.rows;


    static Eigen::MatrixXf J_imu; // 预积分雅克比
    Eigen::VectorXf r_imu;        // 预积分误差
    Eigen::MatrixXf Z_imu;        // 协方差
    J_imu.resize(9, 9);
    init_Jacobian_imu(J_imu);
    r_imu.resize(9);
    Z_imu = Cov_meas_.inverse();

    for (size_t iter = 0; iter != max_iterations_; ++iter)
    {
        H = Matrix6::Zero();
        b = Vector6::Zero();
        // H = Matrix9::Zero();
        // b = Vector9::Zero();
        for (size_t i = 0; i < keypoints.size(); i++)
        {
            // 关键帧像素位置投影到当前帧
            Eigen::Vector3f p = T_curr_update_ * keypoints[i];
            float u = p[0] / p[2] * fx + cx,
                  v = p[1] / p[2] * fy + cy;
            // 当前帧与投影融合得到新像素
            float I_new = evo_utils::interpolate::bilinear(new_img, w, h, u, v); // 双线性插值
            if (I_new == -1.f)
                continue;
            float res = I_new - pixel_values[i]; // 像素值偏差
            if (res >= .95f)
                continue;
            // b.noalias() += 1000 * k.J * res; // 雅克比矩阵
            // H.noalias() += 1000 * k.JJt;     // 海森矩阵
            b.noalias() += J[i] * res; // 雅克比矩阵
            H.noalias() += JJt[i];     // 海森矩阵
        }
        // getError_imu(J_imu, r_imu);
        // b.noalias() += J_imu.transpose() * (Z_imu * r_imu);
        // H.noalias() += J_imu.transpose() * Z_imu * J_imu;
        dx = H.ldlt().solve(b * scale); // 核心位姿更新公式, Cholesky分解
        if ((bool)std::isnan((float)dx[0]))
        {
            LOG(WARNING) << "Matrix close to singular!";
            return;
        }
        updateStateViariant(dx);
        // ------------------------------------------------------
        // imu相关
        // updateStateViariantImu(dx);
        // ------------------------------------------------------
    }
}

void LKSE3::init_Jacobian_imu(Eigen::MatrixXf &J_imu)
{
    /* these are the parts that do not change through iterations */
    J_imu.setZero();
    Eigen::Matrix3f R_i = T_wb_last_.rotation();
    J_imu.block<3, 3>(6, 3) = R_i.transpose();
    J_imu.block<3, 3>(3, 6) = R_i.transpose();
}

void LKSE3::updateStateViariant(Eigen::VectorXf &dx)
{
    T_curr_update_ *= SE3::exp(dx).matrix();
    x_ += dx;
}

void LKSE3::updateStateViariantImu(Eigen::VectorXf &dx)
{
    const Eigen::Vector3f delta_phi = dx.segment<3>(0);
    const Eigen::Matrix3f R_delta = evo_utils::math::exp_SO3(delta_phi);
    const Eigen::Vector3f t_delta = dx.segment<3>(3);
    const Eigen::Vector3f v_delta = dx.segment<3>(6);

    // update T_wb and v
    T_wb_.linear() = T_wb_.rotation() * R_delta;
    T_wb_.translation() += t_delta;
    v_ += v_delta;

    // update T and T_curr
    T_ = T_bc_inv_ * T_wb_ * T_bc_;
    T_curr_ = T_.inverse() * T_kf_;
    x_ += dx;
}

void LKSE3::getError_imu(Eigen::MatrixXf &J_imu, Eigen::VectorXf &r_imu)
{
    Eigen::Matrix3f R_i = T_wb_last_.rotation();    // rotation w.r.t. world frame of last body frame
    Eigen::Matrix3f R_j = T_wb_.rotation();         // rotation w.r.t. world frame of current body frame
    Eigen::Vector3f p_i = T_wb_last_.translation(); // position of last robot frame in world
    Eigen::Vector3f p_j = T_wb_.translation();      // position of current robot frame in world

    // 根据R的误差更新预积分量r
    Eigen::Matrix3f R_err = R_meas_.transpose() * R_i.transpose() * R_j;
    Eigen::Vector3f r_R_imu = evo_utils::math::log_SO3(R_err);
    Eigen::Vector3f r_v_imu = R_i.transpose() * (v_ - v_last_ - g_ * t_meas_) - v_meas_;
    Eigen::Vector3f r_p_imu = R_i.transpose() * (p_j - p_i - v_last_ * t_meas_ - 0.5 * g_ * t_meas_ * t_meas_) - p_meas_;

    r_imu.block<3, 1>(0, 0) = r_R_imu;
    r_imu.block<3, 1>(3, 0) = r_v_imu;
    r_imu.block<3, 1>(6, 0) = r_p_imu;

    J_imu.block<3, 3>(0, 0) = evo_utils::math::log_Jacobian(r_R_imu);
}

void LKSE3::trackFrame()
{
    T_curr_ = T_curr_inv_.inverse();
    T_curr_update_ = T_curr_;
    x_.setZero();

    // v_last_ = v_;
    // T_wb_last_ = T_wb_;
    // T_wb_.linear() = T_wb_last_.rotation() * R_meas_;
    // v_ = T_wb_last_.rotation() * v_meas_ + v_last_ + g_ * t_meas_;
    // T_wb_.translation() = T_wb_last_.rotation() * p_meas_ + T_wb_last_.translation() + v_last_ * t_meas_ + 0.5 * g_ * t_meas_ * t_meas_;
    // T_tmp_ = T_bc_inv_ * T_wb_ * T_bc_;
    // T_curr_ = T_tmp_.inverse() * T_kf_;



    for (size_t lvl = pyramid_levels_; lvl != 0; --lvl)
        updateTransformation(lvl - 1);

    T_curr_inv_ *= SE3::exp(-x_).matrix();
    T_curr_ = T_curr_update_;
    // T_curr_inv_ = T_curr_.inverse();
    T_ = T_kf_ * T_curr_inv_;
    T_wb_ = T_bc_ * T_ * T_bc_inv_;
}

void LKSE3::drawEvents(EventQueue::iterator ev_first, EventQueue::iterator ev_last, cv::Mat &out)
{
    static std::vector<cv::Point> points;
    static std::vector<Eigen::Vector4f> weights;
    if (points.size() == 0)
    {
        cv::Rect rect(0, 0, width_ - 1, height_ - 1);

        for (size_t y = 0; y != height_; ++y)
            for (size_t x = 0; x != width_; ++x)
            {
                cv::Point2d p = c_.rectifyPoint(cv::Point(x, y));
                cv::Point tl(std::floor(p.x), std::floor(p.y));
                Eigen::Vector4f w(0, 0, 0, 0);

                if (rect.contains(tl))
                {
                    const float fx = p.x - tl.x, fy = p.y - tl.y;

                    w[0] = (1.f - fx) * (1.f - fy);
                    w[1] = (fx) * (1.f - fy);
                    w[2] = (1.f - fx) * (fy);
                    w[3] = (fx) * (fy);
                }
                else
                {
                    tl.x = -1;
                }

                points.push_back(tl);
                weights.push_back(w);
            }
    }

    auto draw = [](float &p, const float val)
    { p = std::min(p + val, 1.f); };

    out = cv::Scalar(0);

    for (auto e = ev_first; e != ev_last; ++e)
    {
        const cv::Point &p = points[e->x + e->y * width_];
        if (p.x == -1)
            continue;

        const Eigen::Vector4f &w = weights[e->x + e->y * width_];
        draw(out.at<float>(p.y, p.x), w[0]);
        draw(out.at<float>(p.y, p.x + 1), w[1]);
        draw(out.at<float>(p.y + 1, p.x), w[2]);
        draw(out.at<float>(p.y + 1, p.x + 1), w[3]);
    }
}