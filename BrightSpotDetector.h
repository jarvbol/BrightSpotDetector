#ifndef BRIGHT_SPOT_DETECTOR_H
#define BRIGHT_SPOT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>

class BrightSpotDetector
{
public:
    BrightSpotDetector(
        int crop_size_percent = 28,
        int erode_bbox = 50,
        float min_brightness_factor = 1.3f,
        float min_wh_percent = 1.0f,
        int max_wh_percent = 15,
        int gaussian_sigma = 10);

    int get_crop_size_percent() const;
    int get_erode_bbox() const;
    float get_min_brightness_factor() const;
    float get_min_wh_percent() const;
    int get_max_wh_percent() const;
    int get_gaussian_sigma() const;

    void set_crop_size_percent(int percent);
    void set_erode_bbox(int percent);
    void set_min_brightness_factor(float factor);
    void set_min_wh_percent(float percent);
    void set_max_wh_percent(int percent);
    void set_gaussian_sigma(int sigma);

    cv::Rect get_search_roi(int frame_width, int frame_height) const;

    std::pair<bool, std::optional<std::vector<cv::Rect>>> operator()(
        const cv::Mat &frame,
        const cv::Mat &scope = cv::Mat());

private:
    float gaussian(float sigma, float mean, float x) const;

    int crop_size_percent;
    int erode_bbox;
    float min_brightness_factor;
    float min_wh_percent;
    int max_wh_percent;
    int gaussian_sigma;
};

#endif