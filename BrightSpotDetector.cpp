// #include "BrightSpotDetector.h"
// #include <cmath>
// #include <algorithm>

// BrightSpotDetector::BrightSpotDetector(
//     int crop_size_percent,
//     int erode_bbox,
//     float min_brightness_factor,
//     float min_wh_percent,
//     int max_wh_percent,
//     int gaussian_sigma)
//     : crop_size_percent(crop_size_percent),
//       erode_bbox(erode_bbox),
//       min_brightness_factor(min_brightness_factor),
//       min_wh_percent(min_wh_percent),
//       max_wh_percent(max_wh_percent),
//       gaussian_sigma(gaussian_sigma) {}

// float BrightSpotDetector::gaussian(float sigma, float mean, float x) const
// {
//     float coeff = 1.0f / (sigma * std::sqrt(2.0f * M_PI));
//     float exponent = -((x - mean) * (x - mean)) / (2.0f * sigma * sigma);
//     return coeff * std::exp(exponent);
// }

// std::pair<bool, std::optional<std::vector<cv::Rect>>> BrightSpotDetector::operator()(
//     const cv::Mat &frame,
//     const cv::Mat & /*scope*/)
// {

//     int H = frame.rows;
//     int W = frame.cols;

//     float min_w = (min_wh_percent / 100.0f) * W;
//     float min_h = (min_wh_percent / 100.0f) * H;
//     float max_w = (max_wh_percent / 100.0f) * W;
//     float max_h = (max_wh_percent / 100.0f) * H;

//     int crop_size_w = static_cast<int>((crop_size_percent / 100.0) * W);
//     int crop_size_h = static_cast<int>((crop_size_percent / 100.0) * H);

//     cv::Mat crop = frame(cv::Rect(crop_size_w, crop_size_h,
//                                   W - 2 * crop_size_w,
//                                   H - 2 * crop_size_h));

//     cv::Mat gray_crop;
//     cv::cvtColor(crop, gray_crop, cv::COLOR_BGR2GRAY);

//     cv::Mat binary;
//     cv::threshold(gray_crop, binary, 200, 255, cv::THRESH_BINARY);

//     cv::Mat kernel = cv::Mat::ones(1, 1, CV_8U);
//     cv::Mat opening;
//     cv::morphologyEx(binary, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

//     cv::Mat sure_bg;
//     cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 3);

//     cv::Mat dist_transform;
//     cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5);

//     double maxVal;
//     cv::minMaxLoc(dist_transform, nullptr, &maxVal);
//     cv::Mat sure_fg;
//     cv::threshold(dist_transform, sure_fg, 0.0 * maxVal, 255, cv::THRESH_BINARY);

//     sure_fg.convertTo(sure_fg, CV_8U);
//     cv::Mat unknown = sure_bg - sure_fg;

//     cv::Mat markers;
//     cv::connectedComponents(sure_fg, markers, 8, CV_32S);
//     markers += 1;
//     markers.setTo(cv::Scalar::all(0), unknown == 255);

//     // cv::Mat crop_bgr;
//     // cv::cvtColor(crop, crop_bgr, cv::COLOR_BGR2BGRA);
//     // cv::watershed(crop_bgr, markers);

//     cv::Mat segment_mask = cv::Mat::zeros(crop.size(), CV_8U);
//     segment_mask.setTo(cv::Scalar(255), markers > 1);

//     std::vector<std::vector<cv::Point>> contours;
//     cv::findContours(segment_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

//     std::vector<cv::Rect> valid_bboxes;
//     cv::Point2f center(static_cast<float>(gray_crop.cols) / 2.0f,
//                        static_cast<float>(gray_crop.rows) / 2.0f);

//     for (const auto &cnt : contours)
//     {
//         cv::Rect c_bbox = cv::boundingRect(cnt);

//         if (c_bbox.width < min_w || c_bbox.height < min_h)
//             continue;
//         if (c_bbox.width > max_w || c_bbox.height > max_h)
//             continue;

//         float aspect = std::max(c_bbox.width, c_bbox.height) /
//                        static_cast<float>(std::min(c_bbox.width, c_bbox.height));
//         if (aspect > 10.0f)
//             continue;

//         cv::Mat bbox_crop = gray_crop(c_bbox);
//         double mean_obj = cv::mean(bbox_crop)[0];

//         int pad_w = static_cast<int>(c_bbox.width * (erode_bbox / 100.0));
//         int pad_h = static_cast<int>(c_bbox.height * (erode_bbox / 100.0));

//         int x1 = std::max(0, c_bbox.x - pad_w);
//         int y1 = std::max(0, c_bbox.y - pad_h);
//         int x2 = std::min(gray_crop.cols, c_bbox.x + c_bbox.width + pad_w);
//         int y2 = std::min(gray_crop.rows, c_bbox.y + c_bbox.height + pad_h);

//         cv::Rect outer_rect(x1, y1, x2 - x1, y2 - y1);
//         cv::Mat outer_region = gray_crop(outer_rect);

//         cv::Mat mask = cv::Mat::ones(outer_region.size(), CV_8U);
//         cv::Rect inner_in_outer(c_bbox.x - x1, c_bbox.y - y1, c_bbox.width, c_bbox.height);
//         if (inner_in_outer.area() > 0 && inner_in_outer.x >= 0 && inner_in_outer.y >= 0)
//         {
//             mask(inner_in_outer).setTo(cv::Scalar(0));
//         }

//         cv::Scalar mean_bg_scalar = cv::mean(outer_region, mask);
//         double mean_bg = mean_bg_scalar[0];
//         if (mean_bg <= 1e-5)
//             mean_bg = 1e-5;

//         float brightness_ratio = static_cast<float>(mean_obj / mean_bg);
//         if (brightness_ratio < min_brightness_factor)
//             continue;

//         cv::Point2f bbox_center(c_bbox.x + c_bbox.width / 2.0f,
//                                 c_bbox.y + c_bbox.height / 2.0f);
//         float dist = cv::norm(bbox_center - center);
//         float gauss_weight = gaussian(gaussian_sigma, 0.0f, dist);

//         valid_bboxes.push_back(c_bbox);
//     }

//     if (valid_bboxes.empty())
//     {
//         return {false, std::nullopt};
//     }

//     std::vector<cv::Rect> result_bboxes;
//     for (const auto &bbox : valid_bboxes)
//     {
//         result_bboxes.push_back(cv::Rect(
//             bbox.x + crop_size_w,
//             bbox.y + crop_size_h,
//             bbox.width,
//             bbox.height));
//     }
//     return {true, result_bboxes};
// }

#include "BrightSpotDetector.h"
#include <cmath>
#include <algorithm>
#include <numeric>

BrightSpotDetector::BrightSpotDetector(
    int crop_size_percent,
    int erode_bbox,
    float min_brightness_factor,
    float min_wh_percent,
    int max_wh_percent,
    int gaussian_sigma)
    : crop_size_percent(crop_size_percent),
      erode_bbox(erode_bbox),
      min_brightness_factor(min_brightness_factor),
      min_wh_percent(min_wh_percent),
      max_wh_percent(max_wh_percent),
      gaussian_sigma(gaussian_sigma) {}

float BrightSpotDetector::gaussian(float sigma, float mean, float x) const
{
    float coeff = 1.0f / (sigma * std::sqrt(2.0f * M_PI));
    float exponent = -((x - mean) * (x - mean)) / (2.0f * sigma * sigma);
    return coeff * std::exp(exponent);
}
int BrightSpotDetector::get_crop_size_percent() const
{
    return crop_size_percent;
}
int BrightSpotDetector::get_erode_bbox() const
{
    return erode_bbox;
}
float BrightSpotDetector::get_min_brightness_factor() const
{
    return min_brightness_factor;
}
float BrightSpotDetector::get_min_wh_percent() const
{
    return min_wh_percent;
}
int BrightSpotDetector::get_max_wh_percent() const
{
    return max_wh_percent;
}
int BrightSpotDetector::get_gaussian_sigma() const
{
    return gaussian_sigma;
}

// Сеттеры
void BrightSpotDetector::set_crop_size_percent(int percent)
{
    crop_size_percent = percent;
}
void BrightSpotDetector::set_erode_bbox(int percent)
{
    erode_bbox = percent;
}
void BrightSpotDetector::set_min_brightness_factor(float factor)
{
    min_brightness_factor = factor;
}
void BrightSpotDetector::set_min_wh_percent(float percent)
{
    min_wh_percent = percent;
}
void BrightSpotDetector::set_max_wh_percent(int percent)
{
    max_wh_percent = percent;
}
void BrightSpotDetector::set_gaussian_sigma(int sigma)
{
    gaussian_sigma = sigma;
}

cv::Rect BrightSpotDetector::get_search_roi(int frame_width, int frame_height) const
{
    int crop_w = static_cast<int>((crop_size_percent / 100.0) * frame_width);
    int crop_h = static_cast<int>((crop_size_percent / 100.0) * frame_height);
    return cv::Rect(crop_w, crop_h, frame_width - 2 * crop_w, frame_height - 2 * crop_h);
}

cv::Point2f calc_bbox_center(const cv::Rect &bbox)
{
    return cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
}

std::pair<bool, std::optional<std::vector<cv::Rect>>> BrightSpotDetector::operator()(
    const cv::Mat &frame,
    const cv::Mat & /*scope*/)
{

    int H = frame.rows;
    int W = frame.cols;

    float min_w = (min_wh_percent / 100.0f) * W;
    float min_h = (min_wh_percent / 100.0f) * H;
    float max_w = (max_wh_percent / 100.0f) * W;
    float max_h = (max_wh_percent / 100.0f) * H;

    int crop_size_w = static_cast<int>((crop_size_percent / 100.0) * W);
    int crop_size_h = static_cast<int>((crop_size_percent / 100.0) * H);

    cv::Mat crop = frame(cv::Rect(crop_size_w, crop_size_h,
                                  W - 2 * crop_size_w,
                                  H - 2 * crop_size_h));

    cv::Mat gray_crop;
    cv::cvtColor(crop, gray_crop, cv::COLOR_BGR2GRAY);

    cv::Mat binary;
    cv::threshold(gray_crop, binary, 200, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::Mat::ones(1, 1, CV_8U);
    cv::Mat opening;
    cv::morphologyEx(binary, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

    cv::Mat sure_bg;
    cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 3);

    cv::Mat dist_transform;
    cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5);

    double maxVal;
    cv::minMaxLoc(dist_transform, nullptr, &maxVal);
    cv::Mat sure_fg;
    cv::threshold(dist_transform, sure_fg, 0.0 * maxVal, 255, cv::THRESH_BINARY);

    sure_fg.convertTo(sure_fg, CV_8U);
    cv::Mat unknown = sure_bg - sure_fg;

    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers, 8, CV_32S);
    markers += 1;
    markers.setTo(cv::Scalar::all(0), unknown == 255);

    cv::Mat segment_mask = cv::Mat::zeros(crop.size(), CV_8U);
    segment_mask.setTo(cv::Scalar(255), markers > 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(segment_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> cnt_areas;
    std::vector<float> brightness_scores;

    cv::Point2f center(static_cast<float>(gray_crop.cols) / 2.0f,
                       static_cast<float>(gray_crop.rows) / 2.0f);

    for (const auto &cnt : contours)
    {
        cv::Rect c_bbox = cv::boundingRect(cnt);

        if (c_bbox.width < min_w || c_bbox.height < min_h)
            continue;
        if (c_bbox.width > max_w || c_bbox.height > max_h)
            continue;

        float aspect = std::max(c_bbox.width, c_bbox.height) /
                       static_cast<float>(std::min(c_bbox.width, c_bbox.height));
        if (aspect > 10.0f)
            continue;

        cv::Mat bbox_crop = gray_crop(cv::Rect(c_bbox.x, c_bbox.y, c_bbox.width, c_bbox.height));

        int pad_w = static_cast<int>(c_bbox.width * (erode_bbox / 100.0));
        int pad_h = static_cast<int>(c_bbox.height * (erode_bbox / 100.0));

        int x1 = std::max(0, c_bbox.x - pad_w);
        int y1 = std::max(0, c_bbox.y - pad_h);
        int x2 = std::min(gray_crop.cols, c_bbox.x + c_bbox.width + pad_w);
        int y2 = std::min(gray_crop.rows, c_bbox.y + c_bbox.height + pad_h);

        cv::Rect outer_rect(x1, y1, x2 - x1, y2 - y1);
        cv::Mat outer_region = gray_crop(outer_rect);

        cv::Mat mask = cv::Mat::ones(outer_region.size(), CV_8U);
        cv::Rect inner_in_outer(c_bbox.x - x1, c_bbox.y - y1, c_bbox.width, c_bbox.height);
        if (inner_in_outer.area() > 0 && inner_in_outer.x >= 0 && inner_in_outer.y >= 0)
        {
            mask(inner_in_outer).setTo(cv::Scalar(0));
        }

        cv::Scalar mean_bg_scalar = cv::mean(outer_region, mask);
        double mean_bg = mean_bg_scalar[0];
        if (mean_bg <= 1e-5)
            mean_bg = 1e-5;

        double mean_obj = cv::mean(bbox_crop)[0];
        float brightness_ratio = static_cast<float>(mean_obj / mean_bg);

        if (brightness_ratio < min_brightness_factor)
            continue;

        cv::Point2f bbox_center_point = calc_bbox_center(c_bbox);
        float dist = cv::norm(center - bbox_center_point);
        float score = brightness_ratio * gaussian(gaussian_sigma, 0.0f, dist);

        cnt_areas.push_back(c_bbox);
        brightness_scores.push_back(score);
    }

    if (cnt_areas.empty())
    {
        return {false, std::nullopt};
    }

    std::vector<cv::Rect> result_bboxes;
    for (const auto &bbox : cnt_areas)
    {
        result_bboxes.push_back(cv::Rect(
            bbox.x + crop_size_w,
            bbox.y + crop_size_h,
            bbox.width,
            bbox.height));
    }

    return {true, result_bboxes};
}