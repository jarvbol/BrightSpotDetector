#include "BrightSpotDetector.h"
#include <iostream>

int main()
{
    std::cout << "BrightSpotDetector\n";

    BrightSpotDetector detector;

    cv::Mat frame = cv::imread("/path/to/image");
    if (frame.empty())
    {
        std::cout << "Warning: image not found. Skipping detection.\n";
        return 0;
    }
    detector.set_crop_size_percent(5);

    cv::Rect roi = detector.get_search_roi(frame.cols, frame.rows);
    cv::rectangle(frame, roi, cv::Scalar(0, 0, 255), 2);

    auto [found, boxes] = detector(frame);
    std::cout << "Detected: " << (found ? "YES" : "NO") << "\n";
    if (found && boxes)
    {
        std::cout << "Number of objects: " << boxes->size() << "\n";
        for (const auto &box : *boxes)
        {
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("Result", frame);
        cv::waitKey(0);
    }

    return 0;
}
