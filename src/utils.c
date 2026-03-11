/*
 * utils.c - Utility functions implementation
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <sys/time.h>

#include "utils.h"

// OpenCV headers (optional)
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#endif

struct VideoCapture {
    void* cap;  // OpenCV VideoCapture pointer
    int is_camera;
    Image* current_frame;
};

struct VideoWriter {
    void* writer;  // OpenCV VideoWriter pointer
    int width;
    int height;
};

Image* image_create(int width, int height, int channels) {
    Image* img = (Image*)malloc(sizeof(Image));
    if (!img) return NULL;
    
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (uint8_t*)malloc(width * height * channels);
    
    return img;
}

void image_free(Image* img) {
    if (!img) return;
    if (img->data) free(img->data);
    free(img);
}

VideoCapture* vc_open(const char* path) {
#ifdef USE_OPENCV
    VideoCapture* cap = (VideoCapture*)malloc(sizeof(VideoCapture));
    if (!cap) return NULL;
    
    cap->cap = new cv::VideoCapture(path);
    cap->is_camera = 0;
    cap->current_frame = NULL;
    
    if (!((cv::VideoCapture*)cap->cap)->isOpened()) {
        free(cap);
        return NULL;
    }
    
    return cap;
#else
    (void)path;
    return NULL;
#endif
}

VideoCapture* vc_open_camera(int camera_id) {
#ifdef USE_OPENCV
    VideoCapture* cap = (VideoCapture*)malloc(sizeof(VideoCapture));
    if (!cap) return NULL;
    
    cap->cap = new cv::VideoCapture(camera_id);
    cap->is_camera = 1;
    cap->current_frame = NULL;
    
    if (!((cv::VideoCapture*)cap->cap)->isOpened()) {
        free(cap);
        return NULL;
    }
    
    return cap;
#else
    (void)camera_id;
    return NULL;
#endif
}

Image* vc_read(VideoCapture* cap) {
#ifdef USE_OPENCV
    if (!cap || !cap->cap) return NULL;
    
    cv::Mat frame;
    if (!((cv::VideoCapture*)cap->cap)->read(frame)) {
        return NULL;
    }
    
    // Convert to BGR if needed
    if (frame.channels() == 1) {
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    }
    
    Image* img = image_create(frame.cols, frame.rows, 3);
    if (!img) return NULL;
    
    memcpy(img->data, frame.data, frame.cols * frame.rows * 3);
    
    return img;
#else
    (void)cap;
    return NULL;
#endif
}

void vc_release(VideoCapture* cap) {
#ifdef USE_OPENCV
    if (!cap) return;
    if (cap->cap) {
        delete (cv::VideoCapture*)cap->cap;
    }
    if (cap->current_frame) {
        image_free(cap->current_frame);
    }
    free(cap);
#else
    (void)cap;
#endif
}

VideoWriter* vw_open(const char* path, int width, int height, float fps) {
#ifdef USE_OPENCV
    VideoWriter* writer = (VideoWriter*)malloc(sizeof(VideoWriter));
    if (!writer) return NULL;
    
    writer->width = width;
    writer->height = height;
    
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer->writer = new cv::VideoWriter(path, fourcc, fps, cv::Size(width, height));
    
    if (!((cv::VideoWriter*)writer->writer)->isOpened()) {
        delete (cv::VideoWriter*)writer->writer;
        free(writer);
        return NULL;
    }
    
    return writer;
#else
    (void)path;
    (void)width;
    (void)height;
    (void)fps;
    return NULL;
#endif
}

bool vw_write(VideoWriter* writer, const uint8_t* image_data) {
#ifdef USE_OPENCV
    if (!writer || !writer->writer || !image_data) return false;
    
    cv::Mat frame(writer->height, writer->width, CV_8UC3, (void*)image_data);
    ((cv::VideoWriter*)writer->writer)->write(frame);
    
    return true;
#else
    (void)writer;
    (void)image_data;
    return false;
#endif
}

void vw_close(VideoWriter* writer) {
#ifdef USE_OPENCV
    if (!writer) return;
    if (writer->writer) {
        delete (cv::VideoWriter*)writer->writer;
    }
    free(writer);
#else
    (void)writer;
#endif
}

// Drawing functions (OpenCV-based or fallback)
void draw_box(uint8_t* image, int width, int height, int x1, int y1, int x2, int y2,
              uint8_t r, uint8_t g, uint8_t b, int thickness) {
#ifdef USE_OPENCV
    cv::Mat img(height, width, CV_8UC3, image);
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                 cv::Scalar(b, g, r), thickness);
#else
    // Fallback: simple line drawing
    // Horizontal lines
    for (int x = x1; x <= x2 && x < width; x++) {
        for (int t = 0; t < thickness; t++) {
            int y_top = y1 + t;
            int y_bottom = y2 - t;
            if (y_top >= 0 && y_top < height && x >= 0 && x < width) {
                int idx = (y_top * width + x) * 3;
                image[idx] = b; image[idx+1] = g; image[idx+2] = r;
            }
            if (y_bottom >= 0 && y_bottom < height && x >= 0 && x < width) {
                int idx = (y_bottom * width + x) * 3;
                image[idx] = b; image[idx+1] = g; image[idx+2] = r;
            }
        }
    }
    // Vertical lines
    for (int y = y1; y <= y2 && y < height; y++) {
        for (int t = 0; t < thickness; t++) {
            int x_left = x1 + t;
            int x_right = x2 - t;
            if (y >= 0 && y < height && x_left >= 0 && x_left < width) {
                int idx = (y * width + x_left) * 3;
                image[idx] = b; image[idx+1] = g; image[idx+2] = r;
            }
            if (y >= 0 && y < height && x_right >= 0 && x_right < width) {
                int idx = (y * width + x_right) * 3;
                image[idx] = b; image[idx+1] = g; image[idx+2] = r;
            }
        }
    }
    (void)height;
#endif
}

void draw_text(uint8_t* image, int width, int height, int x, int y,
               const char* text, uint8_t r, uint8_t g, uint8_t b) {
#ifdef USE_OPENCV
    cv::Mat img(height, width, CV_8UC3, image);
    cv::putText(img, text, cv::Point(x, y),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(b, g, r), 1);
#else
    // Simple text rendering not available without OpenCV
    (void)image;
    (void)width;
    (void)height;
    (void)x;
    (void)y;
    (void)text;
    (void)r;
    (void)g;
    (void)b;
#endif
}

void draw_line(uint8_t* image, int width, int height,
               int x1, int y1, int x2, int y2,
               uint8_t r, uint8_t g, uint8_t b, int thickness) {
#ifdef USE_OPENCV
    cv::Mat img(height, width, CV_8UC3, image);
    cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2),
            cv::Scalar(b, g, r), thickness);
#else
    // Bresenham's line algorithm (simplified)
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    
    while (1) {
        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            for (int t = 0; t < thickness; t++) {
                int idx = (y1 * width + x1) * 3;
                image[idx] = b; image[idx+1] = g; image[idx+2] = r;
            }
        }
        
        if (x1 == x2 && y1 == y2) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 < dx) { err += dx; y1 += sy; }
    }
    (void)height;
#endif
}

void draw_circle(uint8_t* image, int width, int height,
                int x, int y, int radius,
                uint8_t r, uint8_t g, uint8_t b) {
#ifdef USE_OPENCV
    cv::Mat img(height, width, CV_8UC3, image);
    cv::circle(img, cv::Point(x, y), radius, cv::Scalar(b, g, r), -1);
#else
    // Simple circle drawing
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx*dx + dy*dy <= radius*radius) {
                int cx = x + dx;
                int cy = y + dy;
                if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                    int idx = (cy * width + cx) * 3;
                    image[idx] = b; image[idx+1] = g; image[idx+2] = r;
                }
            }
        }
    }
    (void)height;
#endif
}

float iou(float x1, float y1, float x2, float y2,
          float x1_, float y1_, float x2_, float y2_) {
    float inter_x1 = (x1 > x1_) ? x1 : x1_;
    float inter_y1 = (y1 > y1_) ? y1 : y1_;
    float inter_x2 = (x2 < x2_) ? x2 : x2_;
    float inter_y2 = (y2 < y2_) ? y2 : y2_;
    
    if (inter_x2 < inter_x1 || inter_y2 < inter_y1) {
        return 0.0f;
    }
    
    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    float area1 = (x2 - x1) * (y2 - y1);
    float area2 = (x2_ - x1_) * (y2_ - y1_);
    float union_area = area1 + area2 - inter_area;
    
    if (union_area < 1e-6f) {
        return 0.0f;
    }
    
    return inter_area / union_area;
}

Image* image_load(const char* path) {
#ifdef USE_OPENCV
    cv::Mat img = cv::imread(path);
    if (img.empty()) return NULL;
    
    if (img.channels() == 1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    
    Image* image = image_create(img.cols, img.rows, 3);
    if (!image) return NULL;
    
    memcpy(image->data, img.data, img.cols * img.rows * 3);
    
    return image;
#else
    (void)path;
    return NULL;
#endif
}

bool image_save(const char* path, const uint8_t* data, int width, int height, int channels) {
#ifdef USE_OPENCV
    cv::Mat img(height, width, CV_8UC3, (void*)data);
    return cv::imwrite(path, img);
#else
    (void)path;
    (void)data;
    (void)width;
    (void)height;
    (void)channels;
    return false;
#endif
}

long long get_timestamp_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void sleep_ms(int ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
}

void log_info(const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("[INFO] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

void log_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("[ERROR] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

void log_debug(const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("[DEBUG] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}
