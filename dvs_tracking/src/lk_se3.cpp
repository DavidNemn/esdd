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

    n_visible_ = 0;
    size_t n_points = 0;
    // 先投影得到kf_img_
    for (const auto &P : map_->points)
    {
        Eigen::Vector3f p(P.x, P.y, P.z);
        p = T_kf_.inverse() * p;
        p[0] = p[0] / p[2] * c_kf_.fx() + c_kf_.cx();
        p[1] = p[1] / p[2] * c_kf_.fy() + c_kf_.cy();
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
        ++n_visible_;
    }

    const int k = map_blur_;
    cv::GaussianBlur(img, img, cv::Size(k, k), 0.);
    cv::GaussianBlur(depthmap, depthmap, cv::Size(k, k), 0.);
    depthmap /= img;
    kf_img_ = img;

    std::nth_element(z_values.begin(), z_values.begin() + z_values.size() / 2, z_values.end());
    depth_median_ = z_values[z_values.size() / 2];
    kf_visibility_ = static_cast<float>(n_visible_) / n_points;

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

    keypoints_.clear();
    // Vector9 vec = Vector9::Zero();
    // Eigen::Map<const Vector9> vec9(&vec(0));
    Vector6 vec = Vector6::Zero();
    Eigen::Map<const Vector6> vec9(&vec(0));

    for (size_t y = 0; y != h; ++y)
    {
        float v = ((float)y - c_kf_.cy()) / c_kf_.fy(); // 实际像素位置v

        for (size_t x = 0; x != w; ++x)
        {
            size_t offset = y * w + x;
            float z = depth_last[offset]; // 深度
            float pixel_value = kf_img_.at<float>(y, x);

            if (pixel_value < .01)
                continue;

            float u = ((float)x - c_kf_.cx()) / c_kf_.fx(); // 实际像素位置u

            float gx = grad_x[offset] * c_kf_.fx(),
                  gy = grad_y[offset] * c_kf_.fy(); // 梯度

            // Vector9 v1, v2;
            // v1 << -1. / z, 0., u / z, u * v, -(1. + u * u), v, 0, 0, 0;
            // v2 << 0., -1. / z, v / z, 1 + v * v, -u * v, -u, 0, 0, 0;
            Vector6 v1, v2;
            v1 << -1. / z, 0., u / z, u * v, -(1. + u * u), v;
            v2 << 0., -1. / z, v / z, 1 + v * v, -u * v, -u;

            vec = gx * v1 + gy * v2; // 雅克比矩阵为6维, J.J^T为6×6

            // 根据上述结果构建Hessian矩阵, 作为属性放进keypoints_里
            keypoints_.push_back(Keypoint(Eigen::Vector3f(u * z, v * z, z), pixel_value, vec, vec9 * vec9.transpose()));
            // keypoints_.push_back(Keypoint(Eigen::Vector3f(u * z, v * z, z), pixel_value));
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
        for (auto i = 0; i != keypoints_.size(); ++i)
        {
            const Keypoint &k = keypoints_[i];
            // 关键帧像素位置投影到当前帧
            Eigen::Vector3f p = T_curr_ * k.P;
            float u = p[0] / p[2] * fx + cx,
                  v = p[1] / p[2] * fy + cy;
            // 当前帧与投影融合得到新像素
            float I_new = evo_utils::interpolate::bilinear(new_img, w, h, u, v); // 双线性插值
            if (I_new == -1.f)
                continue;
            float res = I_new - k.pixel_value; // 像素值偏差
            if (res >= .95f)
                continue;
            // b.noalias() += 1000 * k.J * res; // 雅克比矩阵
            // H.noalias() += 1000 * k.JJt;     // 海森矩阵
            b.noalias() += k.J * res; // 雅克比矩阵
            H.noalias() += k.JJt;     // 海森矩阵
        }
        getError_imu(J_imu, r_imu);
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
    T_curr_ *= SE3::exp(dx).matrix();
    // 这样更新不大行:
    // const Eigen::Vector3f delta_phi = dx.segment<3>(0);
    // const Eigen::Matrix3f R_delta = evo_utils::math::exp_SO3(delta_phi);
    // const Eigen::Vector3f t_delta = dx.segment<3>(3);
    // T_curr_.linear() = T_curr_.rotation() * R_delta;
    // T_curr_.translation() += t_delta;
    x_ += dx;
    // const Eigen::Vector3f delta_phi = dx.segment<3>(0);
    // const Eigen::Matrix3f R_delta = evo_utils::math::exp_SO3(delta_phi);
    // const Eigen::Vector3f t_delta = dx.segment<3>(3);

    // // update T_wb and v
    // T_wb_.linear() = T_wb_.rotation() * R_delta;
    // T_wb_.translation() += t_delta;

    // x_ += dx;
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
    // T_curr_ = T_curr_inv_.inverse();
    x_.setZero();

    v_last_ = v_;
    T_wb_last_ = T_wb_;
    T_wb_.linear() = T_wb_last_.rotation() * R_meas_;
    v_ = T_wb_last_.rotation() * v_meas_ + v_last_ + g_ * t_meas_;
    T_wb_.translation() = T_wb_last_.rotation() * p_meas_ + T_wb_last_.translation() + v_last_ * t_meas_ + 0.5 * g_ * t_meas_ * t_meas_;
    T_tmp_ = T_bc_inv_ * T_wb_ * T_bc_;
    T_curr_ = T_tmp_.inverse() * T_kf_;

    for (size_t lvl = pyramid_levels_; lvl != 0; --lvl)
        updateTransformation(lvl - 1);
    T_curr_inv_ *= SE3::exp(-x_).matrix();
    T_ = T_kf_ * T_curr_inv_; // 发布的就是T_, 所以不能放到updateTransformation里边
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