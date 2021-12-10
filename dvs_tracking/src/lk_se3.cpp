#include "dvs_tracking/lk_se3.hpp"

#include <glog/logging.h>

#include <boost/range/irange.hpp>
#include <random>

#include "dvs_tracking/weight_functions.hpp"
#include "evo_utils/interpolation.hpp"

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

            float gx = grad_x[offset],
                  gy = grad_y[offset]; // 梯度

            Vector6 v1, v2;
            v1 << -fx_ * Z_inv, 0., fx_ * X * Z2_inv, fx_ * X * Y * Z2_inv, -(fx_ + fx_ * X * X * Z2_inv), fx_ * Y * Z_inv;
            v2 << 0, -fy_ * Z_inv, fy_ * Y * Z2_inv, fy_ + fy_ * Y * Y * Z2_inv, -fy_ * X * Y * Z2_inv, -fy_ * X * Z_inv;

            vec = gx * v1 + gy * v2; // 雅克比矩阵为6维, J.J^T为6×6

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
    for (size_t iter = 0; iter != max_iterations_; ++iter)
    {
        H = Matrix6::Zero();
        b = Vector6::Zero();
        for (size_t i = 0; i < keypoints.size(); i++)
        {
            // 关键帧像素位置投影到当前帧
            Eigen::Vector3f p = T_curr_ * keypoints[i];
            float u = p[0] / p[2] * fx + cx,
                  v = p[1] / p[2] * fy + cy;
            // 当前帧与投影融合得到新像素
            float I_new = evo_utils::interpolate::bilinear(new_img, w, h, u, v); // 双线性插值
            if (I_new == -1.f)
                continue;
            float res = I_new - pixel_values[i];
            if (res >= .95f)
                continue;
            b.noalias() += J[i] * res; // 雅克比矩阵
            H.noalias() += JJt[i];     // 海森矩阵
        }
        dx = H.ldlt().solve(b * scale); // 核心位姿更新公式, Cholesky分解
        if ((bool)std::isnan((float)dx[0]))
        {
            LOG(WARNING) << "Matrix close to singular!";
            return;
        }
        T_curr_ *= SE3::exp(dx).matrix();
        x_ += dx;
    }
}

void LKSE3::trackFrame()
{
    cv::buildPyramid(new_img_, pyr_new_, pyramid_levels_); // 构建图像金字塔
    T_curr_ = T_curr_inv_.inverse();
    x_.setZero();

    for (size_t lvl = pyramid_levels_; lvl != 0; --lvl)
        updateTransformation(lvl - 1);

    T_curr_inv_ *= SE3::exp(-x_).matrix();
    T_ = T_kf_ * T_curr_inv_;
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