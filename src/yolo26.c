/*
 * yolo26.c - YOLO26 inference using ONNX Runtime
 * 
 * Compatible with ONNX Runtime 1.x C API
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "yolo26.h"

// ONNX Runtime C API
#include <onnxruntime_c_api.h>

struct Yolo26Model {
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* session_options;
    
    char* input_name;
    char* output_name;
    
    float conf_threshold;
    int num_classes;
    int input_width;
    int input_height;
    
    // Preprocessing buffer
    uint8_t* resize_buffer;
    float* input_tensor;
    
    // ONNX API
    const OrtApi* api;
};

static const char* get_error_message(const OrtApi* api, OrtStatus* status) {
    if (status == NULL) {
        return "Unknown error";
    }
    return api->GetErrorMessage(status);
}

#define CHECK_STATUS(api, expr) \
    do { \
        OrtStatus* status = (expr); \
        if (status != NULL) { \
            fprintf(stderr, "ONNX Error: %s\n", get_error_message(api, status)); \
            api->ReleaseStatus(status); \
            return NULL; \
        } \
    } while(0)

#define CHECK_STATUS_NULL(api, expr) \
    do { \
        OrtStatus* status = (expr); \
        if (status != NULL) { \
            fprintf(stderr, "ONNX Error: %s\n", get_error_message(api, status)); \
            api->ReleaseStatus(status); \
            return NULL; \
        } \
    } while(0)

static void letterbox_resize(const uint8_t* src, uint8_t* dst,
                            int src_w, int src_h, int dst_w, int dst_h) {
    float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    
    // Bilinear interpolation
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float src_x = (x + 0.5f) / scale - 0.5f;
            float src_y = (y + 0.5f) / scale - 0.5f;
            
            int x0 = (int)floorf(src_x);
            int y0 = (int)floorf(src_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            // Clamp
            x0 = (x0 < 0) ? 0 : (x0 >= src_w) ? src_w - 1 : x0;
            y0 = (y0 < 0) ? 0 : (y0 >= src_h) ? src_h - 1 : y0;
            x1 = (x1 < 0) ? 0 : (x1 >= src_w) ? src_w - 1 : x1;
            y1 = (y1 < 0) ? 0 : (y1 >= src_h) ? src_h - 1 : y1;
            
            // Bilinear weights
            float wx1 = src_x - x0;
            float wy1 = src_y - y0;
            float wx0 = 1.0f - wx1;
            float wy0 = 1.0f - wy1;
            
            for (int c = 0; c < 3; c++) {
                float v00 = src[y0 * src_w * 3 + x0 * 3 + c];
                float v01 = src[y0 * src_w * 3 + x1 * 3 + c];
                float v10 = src[y1 * src_w * 3 + x0 * 3 + c];
                float v11 = src[y1 * src_w * 3 + x1 * 3 + c];
                
                float value = wx0 * (wy0 * v00 + wy1 * v10) +
                             wx1 * (wy0 * v01 + wy1 * v11);
                
                dst[y * dst_w * 3 + x * 3 + c] = (uint8_t)(value + 0.5f);
            }
        }
    }
}

static void normalize_image(uint8_t* src, float* dst, int width, int height) {
    for (int i = 0; i < width * height * 3; i++) {
        dst[i] = src[i] / 255.0f;
    }
}

Yolo26Model* yolo26_init(const char* onnx_model_path, float conf_threshold, int num_classes) {
    // Get ONNX Runtime API
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api == NULL) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return NULL;
    }
    
    // Create environment
    OrtEnv* env = NULL;
    CHECK_STATUS(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "YOLO26", &env));
    
    // Create session options
    OrtSessionOptions* session_options = NULL;
    CHECK_STATUS(api, api->CreateSessionOptions(&session_options));
    
    // Optimize for performance
    CHECK_STATUS(api, api->SetGraphOptimizationLevel(
        session_options, GRAPH_OPTIMIZATION_LEVEL_ENABLE));
    
    // Create session
    OrtSession* session = NULL;
    CHECK_STATUS(api, api->CreateSession(env, onnx_model_path, session_options, &session));
    
    // Get allocator
    OrtAllocator* allocator = NULL;
    CHECK_STATUS(api, api->GetDefaultAllocator(&allocator));
    
    // Get input name
    char* input_name = NULL;
    CHECK_STATUS(api, api->SessionGetInputName(session, 0, allocator, &input_name));
    
    // Get output name
    char* output_name = NULL;
    CHECK_STATUS(api, api->SessionGetOutputName(session, 0, allocator, &output_name));
    
    // Get input shape info
    OrtTypeInfo* input_type_info = NULL;
    CHECK_STATUS(api, api->SessionGetInputTypeInfo(session, 0, &input_type_info));
    
    const OrtTensorTypeAndShapeInfo* input_tensor_info = NULL;
    CHECK_STATUS(api, api->CastTypeInfoToTensorInfo(input_type_info, &input_tensor_info));
    
    size_t input_dims_count = 0;
    CHECK_STATUS(api, api->GetDimensions(input_tensor_info, NULL, &input_dims_count));
    
    int64_t* input_dims = (int64_t*)malloc(input_dims_count * sizeof(int64_t));
    CHECK_STATUS(api, api->GetDimensions(input_tensor_info, input_dims, input_dims_count));
    
    // Default input size
    int input_width = 640;
    int input_height = 640;
    
    if (input_dims_count >= 4) {
        input_height = (int)input_dims[2];
        input_width = (int)input_dims[3];
    }
    
    api->ReleaseTypeInfo(input_type_info);
    free(input_dims);
    
    // Create model structure
    Yolo26Model* model = (Yolo26Model*)malloc(sizeof(Yolo26Model));
    if (!model) {
        return NULL;
    }
    
    model->api = api;
    model->env = env;
    model->session = session;
    model->session_options = session_options;
    model->input_name = input_name;
    model->output_name = output_name;
    model->conf_threshold = conf_threshold;
    model->num_classes = num_classes;
    model->input_width = input_width;
    model->input_height = input_height;
    
    // Allocate buffers
    model->resize_buffer = (uint8_t*)malloc(input_width * input_height * 3);
    model->input_tensor = (float*)malloc(1 * 3 * input_height * input_width * sizeof(float));
    
    return model;
}

YoloDetections* yolo26_detect(Yolo26Model* model, const uint8_t* image_data,
                             int width, int height, int channels) {
    if (!model || !image_data) {
        return NULL;
    }
    
    const OrtApi* api = model->api;
    
    // Preprocess: resize and normalize
    letterbox_resize(image_data, model->resize_buffer,
                    width, height, model->input_width, model->input_height);
    normalize_image(model->resize_buffer, model->input_tensor,
                   model->input_width, model->input_height);
    
    // Create input tensor
    int64_t input_shape[] = {1, 3, model->input_height, model->input_width};
    
    OrtValue* input_tensor = NULL;
    CHECK_STATUS_NULL(api, api->CreateTensorWithDataAsOrtValue(
        model->env,
        model->input_tensor,
        1 * 3 * model->input_height * model->input_width * sizeof(float),
        input_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    ));
    
    // Run inference
    const char* input_names[] = {model->input_name};
    const char* output_names[] = {model->output_name};
    
    OrtValue* output_tensor = NULL;
    CHECK_STATUS_NULL(api, api->Run(
        model->session,
        NULL,
        input_names,
        &input_tensor,
        1,
        output_names,
        1,
        &output_tensor
    ));
    
    // Get output data
    float* output_data = NULL;
    CHECK_STATUS_NULL(api, api->GetTensorData(output_tensor, (void**)&output_data));
    
    // Get output shape
    OrtTypeInfo* output_type_info = NULL;
    CHECK_STATUS_NULL(api, api->SessionGetOutputTypeInfo(model->session, 0, &output_type_info));
    
    const OrtTensorTypeAndShapeInfo* output_tensor_info = NULL;
    CHECK_STATUS_NULL(api, api->CastTypeInfoToTensorInfo(output_type_info, &output_tensor_info));
    
    size_t output_dims_count = 0;
    CHECK_STATUS_NULL(api, api->GetDimensions(output_tensor_info, NULL, &output_dims_count));
    
    int64_t* output_dims = (int64_t*)malloc(output_dims_count * sizeof(int64_t));
    CHECK_STATUS_NULL(api, api->GetDimensions(output_tensor_info, output_dims, output_dims_count));
    
    // Parse outputs: [batch, num_predictions, (x, y, w, h, obj, cls1, cls2, ...)]
    size_t num_predictions = output_dims[1];
    
    // Create detections
    YoloDetections* results = (YoloDetections*)malloc(sizeof(YoloDetections));
    results->capacity = num_predictions;
    results->detections = (YoloDetection*)malloc(sizeof(YoloDetection) * num_predictions);
    results->count = 0;
    
    // Scale factors
    float scale_x = (float)width / model->input_width;
    float scale_y = (float)height / model->input_height;
    
    for (size_t i = 0; i < num_predictions; i++) {
        float* pred = &output_data[i * (5 + model->num_classes)];
        
        float obj_conf = pred[4];
        if (obj_conf < model->conf_threshold) {
            continue;
        }
        
        // Find class with highest probability
        int class_id = 0;
        float class_conf = pred[5];
        for (int c = 1; c < model->num_classes; c++) {
            if (pred[5 + c] > class_conf) {
                class_conf = pred[5 + c];
                class_id = c;
            }
        }
        
        // Final confidence
        float final_conf = obj_conf * class_conf;
        if (final_conf < model->conf_threshold) {
            continue;
        }
        
        // Convert from center to corner format
        float cx = pred[0] * scale_x;
        float cy = pred[1] * scale_y;
        float w = pred[2] * scale_x;
        float h = pred[3] * scale_y;
        
        float x1 = cx - w / 2;
        float y1 = cy - h / 2;
        float x2 = cx + w / 2;
        float y2 = cy + h / 2;
        
        // Clamp to image bounds
        x1 = (x1 < 0) ? 0 : (x1 >= width) ? width - 1 : x1;
        y1 = (y1 < 0) ? 0 : (y1 >= height) ? height - 1 : y1;
        x2 = (x2 < 0) ? 0 : (x2 >= width) ? width - 1 : x2;
        y2 = (y2 < 0) ? 0 : (y2 >= height) ? height - 1 : y2;
        
        // Add detection
        results->detections[results->count].x1 = x1;
        results->detections[results->count].y1 = y1;
        results->detections[results->count].x2 = x2;
        results->detections[results->count].y2 = y2;
        results->detections[results->count].confidence = final_conf;
        results->detections[results->count].class_id = class_id;
        results->detections[results->count].track_id = -1;
        
        results->count++;
    }
    
    // Cleanup
    free(output_dims);
    api->ReleaseValue(output_tensor);
    api->ReleaseValue(input_tensor);
    api->ReleaseTypeInfo(output_type_info);
    
    return results;
}

void yolo26_free_detections(YoloDetections* detections) {
    if (!detections) {
        return;
    }
    
    if (detections->detections) {
        free(detections->detections);
    }
    free(detections);
}

void yolo26_destroy(Yolo26Model* model) {
    if (!model) {
        return;
    }
    
    if (model->resize_buffer) free(model->resize_buffer);
    if (model->input_tensor) free(model->input_tensor);
    if (model->input_name) model->api->ReleaseAllocator(model->input_name);
    if (model->output_name) model->api->ReleaseAllocator(model->output_name);
    if (model->session) model->api->ReleaseSession(model->session);
    if (model->session_options) model->api->ReleaseSessionOptions(model->session_options);
    if (model->env) model->api->ReleaseEnv(model->env);
    
    free(model);
}

const char* yolo26_get_name(Yolo26Model* model) {
    (void)model;
    return "YOLO26";
}
