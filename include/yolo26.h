/*
 * yolo26.h - YOLO26 object detection header
 */

#ifndef YOLO26_H
#define YOLO26_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Detection result structure
typedef struct {
    float x1, y1, x2, y2;   // Bounding box
    float confidence;        // Detection confidence
    int class_id;           // Class ID
    int track_id;           // Track ID (-1 if not tracked)
} YoloDetection;

// Detection array
typedef struct {
    YoloDetection* detections;
    size_t count;
    size_t capacity;
} YoloDetections;

// YOLO26 model handle
typedef struct Yolo26Model Yolo26Model;

// Initialize YOLO26 model from ONNX file
Yolo26Model* yolo26_init(const char* onnx_model_path, float conf_threshold, int num_classes);

// Run inference on image
YoloDetections* yolo26_detect(Yolo26Model* model, const uint8_t* image_data, 
                               int width, int height, int channels);

// Free detection results
void yolo26_free_detections(YoloDetections* detections);

// Free YOLO model
void yolo26_destroy(Yolo26Model* model);

// Get model name
const char* yolo26_get_name(Yolo26Model* model);

#ifdef __cplusplus
}
#endif

#endif // YOLO26_H
