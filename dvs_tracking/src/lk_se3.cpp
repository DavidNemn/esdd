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
    Vector6 vec = Vector6::Zero();
    Eigen::Map<const Vector6> vec6(&vec(0));

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

            Vector6 v1, v2;
            v1 << -1. / z, 0., u / z, u * v, -(1. + u * u), v;
            v2 << 0., -1. / z, v / z, 1 + v * v, -u * v, -u;

            vec = gx * v1 + gy * v2; // 雅克比矩阵为6维, J.J^T为6×6

            // 根据上述结果构建Hessian矩阵, 作为属性放进keypoints_里
            keypoints_.push_back(Keypoint(Eigen::Vector3f(u * z, v * z, z), pixel_value, vec, vec6 * vec6.transpose()));
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
    for (size_t iter = 0; iter != max_iterations_; ++iter)
    {
        H = Matrix6::Zero();
        b = Vector6::Zero();
        for (auto i = 0; i != keypoints_.size(); ++i)
        {
            const Keypoint &k = keypoints_[i];
            // 关键帧像素位置投影到当前帧
            Eigen::Vector3f p = T_curr_inv_ * k.P;
            float u = p[0] / p[2] * fx + cx,
                  v = p[1] / p[2] * fy + cy;
            // 当前帧与投影融合得到新像素
            float I_new = evo_utils::interpolate::bilinear(new_img, w, h, u, v); // 双线性插值
            if (I_new == -1.f)
                continue;
            float res = I_new - k.pixel_value; // 像素值偏差
            if (res >= .95f)
                continue;
            b.noalias() += k.J * res; // 雅克比矩阵
            H.noalias() += k.JJt;     // 海森矩阵
        }
        dx = H.ldlt().solve(b * scale); // 核心位姿更新公式, Cholesky分解
        if ((bool)std::isnan((float)dx[0]))
        {
            LOG(WARNING) << "Matrix close to singular!";
            return;
        }
        T_curr_inv_ *= SE3::exp(dx).matrix();
        x_ += dx;
    }
}

void LKSE3::trackFrame()
{
    T_curr_inv_ = T_curr_.inverse();
    x_.setZero();

    for (size_t lvl = pyramid_levels_; lvl != 0; --lvl)
        updateTransformation(lvl - 1);

    T_curr_ *= SE3::exp(-x_).matrix();
    T_ = T_kf_ * T_curr_;
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