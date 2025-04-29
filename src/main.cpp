#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat sequentialBlur(const Mat& input) {
    Mat output = input.clone();
    // проверка размера
    if (input.rows < 3 || input.cols < 3) {
        std::cerr << "Image is too small for 3x3 blur" << std::endl;
        return output;
    }

    for (int y = 1; y < input.rows - 1; y+=1) {
        for (int x = 1; x < input.cols - 1; x+=1) {
            Vec3i sum(0, 0, 0); // Используем int для суммы так как может быть переполнение
            for (int dy = -1; dy <= 1; dy+=1) {
                for (int dx = -1; dx <= 1; dx+=1) {
                    Vec3b pixel = input.at<Vec3b>(y + dy, x + dx);
                    sum[0] += pixel[0];
                    sum[1] += pixel[1];
                    sum[2] += pixel[2];
                }
            }
            // Округление +4 до целого
            output.at<Vec3b>(y, x) = Vec3b( (sum[0] + 4) / 9, (sum[1] + 4) / 9, (sum[2] + 4) / 9);
        }
    }
    return output;
}

Mat parallelBlurThreads(const Mat& input, int num_threads = 4) {
    Mat output = input.clone();
    std::vector<std::thread> threads;
    int strip_height = input.rows / num_threads;

    auto processStrip = [&](int start_y, int end_y) {
        start_y = std::max(1, start_y);
        end_y = std::min(input.rows - 1, end_y);
        for (int y = start_y; y < end_y; y+=1) {
            for (int x = 1; x < input.cols - 1; x+=1) {
                Vec3i sum(0, 0, 0); // Используем int для суммы так как может быть переполнение
                for (int dy = -1; dy <= 1; dy+=1) {
                    for (int dx = -1; dx <= 1; dx+=1) {
                        Vec3b pixel = input.at<Vec3b>(y + dy, x + dx);
                        sum[0] += pixel[0];
                        sum[1] += pixel[1];
                        sum[2] += pixel[2];
                    }
                }
                // Округление +4 до целого
                output.at<Vec3b>(y, x) = Vec3b( (sum[0] + 4) / 9, (sum[1] + 4) / 9, (sum[2] + 4) / 9);
            }
        }
    };

    // Запуск потков
    for (int i = 0; i < num_threads; i+=1) {
        int start_y = i * strip_height;
        int end_y = (i == num_threads - 1) ? input.rows : (i + 1) * strip_height;
        threads.emplace_back(processStrip, start_y, end_y);
    }

    // Ожидание завершения потоков
    for (auto& t : threads) {
        t.join();
    }

    return output;
}

void atomic_mutex(int num_threads = 4, int iterations = 1000000) {
    std::atomic<int> counter_atomic{0};
    int counter_mutex = 0;
    std::mutex mtx;

    // Функция для atomic
    auto Atomic = [&]() {
        for (int i = 0; i < iterations; i+=1) {
            counter_atomic++;
        }
    };

    // Функция для мьютекса
    auto Mutex = [&]() {
        for (int i = 0; i < iterations; i+=1) {
            std::lock_guard<std::mutex> lock(mtx);
            counter_mutex++;
        }
    };
    // atomic
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i+=1) {
        threads.emplace_back(Atomic);
    }
    for (auto& t : threads) { // не даём работать main
        t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Atomic time: " << diff.count() << "s, Result: " << counter_atomic << std::endl;

    // мьютекс
    threads.clear();
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; i+=1) {
        threads.emplace_back(Mutex);
    }
    for (auto& t : threads) { // не даём работать main
        t.join();
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Mutex time: " << diff.count() << "s, Result: " << counter_mutex << std::endl;
}

int main() {
    // Загрузка через OpenCV
    Mat input = imread("/Users/maksimkuznetsov/CLionProjects/2sem_laba5/images/Example.jpg");
    if (input.empty()) {
        std::cerr << "Cant open image" << std::endl;
        return -1;
    }

    // Последовательное
    auto start = std::chrono::high_resolution_clock::now();
    Mat output_seq = sequentialBlur(input);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "sequentialBlur time: " << diff.count() << "s\n";
    imwrite("/Users/maksimkuznetsov/CLionProjects/2sem_laba5/images/output_seq.jpg", output_seq);

    // параллельно
    start = std::chrono::high_resolution_clock::now();
    Mat output_par = parallelBlurThreads(input);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "parallelBlurThreads time: " << diff.count() << "s\n";
    imwrite("/Users/maksimkuznetsov/CLionProjects/2sem_laba5/images/output_par.jpg", output_par);
    // атомарные и мьютекс
    atomic_mutex();

    return 0;
}