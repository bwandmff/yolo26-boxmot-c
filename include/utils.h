/*
 * utils.h - Utility functions header
 */

#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Image structure
typedef struct {
    uint8_t* data;
    int width;
    int height;
    int channels;
} Image;

// Video capture handle (abstract)
typedef struct VideoCapture VideoCapture;

// Initialize video capture from file or camera
VideoCapture* vc_open(const char* path);

VideoCapture* vc_open_camera(int camera_id);

// Get next frame (returns NULL when no more frames)
Image* vc_read(VideoCapture* cap);

// Release video capture
void vc_release(VideoCapture* cap);

// Free image
void image_free(Image* img);

// Video writer
typedef struct VideoWriter VideoWriter;

// Create video writer
VideoWriter* vw_open(const char* path, int width, int height, float fps);

// Write frame
bool vw_write(VideoWriter* writer, const uint8_t* image_data);

// Close writer
void vw_close(VideoWriter* writer);

// Drawing functions
void draw_box(uint8_t* image, int width, int height, int x1, int y1, int x2, int y2, 
              uint8_t r, uint8_t g, uint8_t b, int thickness);

void draw_text(uint8_t* image, int width, int height, int x, int y, 
               const char* text, uint8_t r, uint8_t g, uint8_t b);

void draw_line(uint8_t* image, int width, int height, 
               int x1, int y1, int x2, int y2,
               uint8_t r, uint8_t g, uint8_t b, int thickness);

void draw_circle(uint8_t* image, int width, int height, 
                 int x, int y, int radius,
                 uint8_t r, uint8_t g, uint8_t b);

// IOU calculation
float iou(float x1, float y1, float x2, float y2,
          float x1_, float y1_, float x2_, float y2_);

// Load image from file (using stb_image)
Image* image_load(const char* path);

// Save image to file
bool image_save(const char* path, const uint8_t* data, int width, int height, int channels);

// Get current timestamp in milliseconds
long long get_timestamp_ms(void);

// Sleep for milliseconds
void sleep_ms(int ms);

// Print log message
void log_info(const char* format, ...);
void log_error(const char* format, ...);
void log_debug(const char* format, ...);

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
