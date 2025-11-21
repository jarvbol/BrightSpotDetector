# BrightSpotDetector
A C++ implementation of a bright spot detector.

---

## Features

- Detects bright spots in grayscale images (e.g., thermal/IR images)
- Configurable detection parameters
- Adjustable search ROI (region of interest)
- Returns all detected bounding boxes
- Based on OpenCV
- Works on Windows, Linux, macOS

---

## Requirements

- **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **OpenCV 4.x**
- **pkg-config** (Linux/macOS) — для автоматического поиска библиотек
- **CMake 3.16+** (optional, for building with CMake)

---

## Building

macOS/Linux:
```bash
g++ -std=c++17 -Wall -Wextra -g3 main.cpp BrightSpotDetector.cpp -o detector `pkg-config --cflags --libs opencv4`
```

Windows (MinGW/MSYS2):
```bash
g++ -std=c++17 -Wall -Wextra -g3 main.cpp BrightSpotDetector.cpp -o detector.exe `pkg-config --cflags --libs opencv4`
```

---

## Usage

1. Place your image as test.png in the project root.
2. Run the executable:
macOS/Linux:
```bash
./detector
```
    
Windows (MinGW/MSYS2):
```bash
detector.exe
```

---

## Adjusting Parameters

| Parameter |	Type | Default | Description |
|---------|--------|--------|--------|
| `crop_size_percent`	| int	| 28	| Percentage of image borders to exclude from search (0–49) |
| `erode_bbox`	| int |	50 |	Percentage to expand bbox for background brightness calculation |
| `min_brightness_factor`	| float	| 1.3 |	Minimum brightness ratio to be considered a bright spot |
| `min_wh_percent`	| float	| 1.0 |	Minimum object width/height as % of frame |
| `max_wh_percent`	| int	| 15 |	Maximum object width/height as % of frame |
| `gaussian_sigma`	| int	| 10 |	Gaussian penalty for distance from center |

---

## Example of the operation



---

## License

MIT — feel free to use, modify, and distribute.



