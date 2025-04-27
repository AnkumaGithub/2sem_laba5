#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat sequentialBlur(const Mat& input) {
    Mat output = input.clone();
    for (int y = 1; y < input.rows - 1; ++y) {
        for (int x = 1; x < input.cols - 1; ++x) {
            Vec3b sum(0, 0, 0);
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    sum += input.at<Vec3b>(y + dy, x + dx);
                }
            }
            output.at<Vec3b>(y, x) = sum / 9;
        }
    }
    return output;
}

Mat parallelBlurThreads(const Mat& input, int num_threads = 4) {
    Mat output = input.clone();
    std::vector<std::thread> threads;
    int strip_height = input.rows / num_threads;

    auto processStrip = [&](int start_y, int end_y) {
        for (int y = start_y; y < end_y; ++y) {
            for (int x = 1; x < input.cols - 1; ++x) {
                Vec3b sum(0, 0, 0);
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        sum += input.at<Vec3b>(y + dy, x + dx);
                    }
                }
                output.at<Vec3b>(y, x) = sum / 9;
            }
        }
    };
}

void atomicExample() {
    std::atomic<int> counter_atomic{0};
    int counter_mutex = 0;
    std::mutex mtx;

    auto incrementAtomic = [&]() {
        for (int i = 0; i < 1000000; ++i) counter_atomic++;
    };

    auto incrementMutex = [&]() {
        for (int i = 0; i < 1000000; ++i) {
            std::lock_guard<std::mutex> lock(mtx);
            counter_mutex++;
        }
    };
}

int main() {
    // Загрузка изображения через OpenCV
    Mat input = imread("images/input.jpg");
    if (input.empty()) {
        std::cerr << "Could not open image!" << std::endl;
        return -1;
    }

    // Последовательное размытие
    auto start = std::chrono::high_resolution_clock::now();
    Mat output_seq = sequentialBlur(input);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Sequential blur time: " << diff.count() << "s\n";
    imwrite("images/output_seq.jpg", output_seq);

    // параллельно
    start = std::chrono::high_resolution_clock::now();
    Mat output_par = parallelBlurThreads(input);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Parallel blur time: " << diff.count() << "s\n";
    imwrite("images/output_par.jpg", output_par);

    atomicExample();
    return 0;
}