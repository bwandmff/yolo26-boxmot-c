/*
 * main.c - YOLO26 + ByteTrack Linux Application
 * 
 * Compile: make
 * Run: ./yolo26_bytetrack --model models/yolo26n.onnx --source video.mp4 --output output.mp4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <signal.h>

#include "yolo26.h"
#include "bytetrack.h"
#include "utils.h"

// Global flags for signal handling
static volatile int g_running = 1;

void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
    log_info("Received signal, shutting down...");
}

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --model <path>       ONNX model path (default: models/yolo26n.onnx)\n");
    printf("  --source <path>       Input video or camera id (default: 0)\n");
    printf("  --output <path>      Output video path (default: output.mp4)\n");
    printf("  --tracker <name>     Tracker: bytetrack, botsort, ocsort (default: bytetrack)\n");
    printf("  --conf <float>       Confidence threshold (default: 0.5)\n");
    printf("  --show               Display real-time window\n");
    printf("  --trail-length <n>   Trail length for visualization (default: 30)\n");
    printf("  --help               Show this help\n");
}

int main(int argc, char* argv[]) {
    // Default parameters
    char model_path[512] = "models/yolo26n.onnx";
    char source_path[512] = "0";
    char output_path[512] = "output.mp4";
    char tracker_name[32] = "bytetrack";
    float conf_threshold = 0.5f;
    int show_window = 0;
    int trail_length = 30;
    
    // Parse arguments
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"source", required_argument, 0, 's'},
        {"output", required_argument, 0, 'o'},
        {"tracker", required_argument, 0, 't'},
        {"conf", required_argument, 0, 'c'},
        {"show", no_argument, 0, 'd'},
        {"trail-length", required_argument, 0, 'l'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "m:s:o:t:c:dl:h", 
                              long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm':
                strncpy(model_path, optarg, sizeof(model_path) - 1);
                break;
            case 's':
                strncpy(source_path, optarg, sizeof(source_path) - 1);
                break;
            case 'o':
                strncpy(output_path, optarg, sizeof(output_path) - 1);
                break;
            case 't':
                strncpy(tracker_name, optarg, sizeof(tracker_name) - 1);
                break;
            case 'c':
                conf_threshold = atof(optarg);
                break;
            case 'd':
                show_window = 1;
                break;
            case 'l':
                trail_length = atoi(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    log_info("========================================");
    log_info("YOLO26 + ByteTrack Tracking for Linux");
    log_info("========================================");
    log_info("Model: %s", model_path);
    log_info("Source: %s", source_path);
    log_info("Output: %s", output_path);
    log_info("Tracker: %s", tracker_name);
    log_info("Confidence: %.2f", conf_threshold);
    log_info("========================================");
    
    // Open video source
    VideoCapture* cap = NULL;
    if (strlen(source_path) == 1 && source_path[0] >= '0' && source_path[0] <= '9') {
        cap = vc_open_camera(source_path[0] - '0');
    } else {
        cap = vc_open(source_path);
    }
    
    if (!cap) {
        log_error("Failed to open video source: %s", source_path);
        return 1;
    }
    
    // Read first frame to get dimensions
    Image* first_frame = vc_read(cap);
    if (!first_frame) {
        log_error("Failed to read first frame");
        vc_release(cap);
        return 1;
    }
    
    int width = first_frame->width;
    int height = first_frame->height;
    log_info("Video: %dx%d", width, height);
    
    // Initialize YOLO26 model
    log_info("Loading YOLO26 model...");
    Yolo26Model* yolo = yolo26_init(model_path, conf_threshold, 80);  // COCO has 80 classes
    if (!yolo) {
        log_error("Failed to load YOLO model");
        image_free(first_frame);
        vc_release(cap);
        return 1;
    }
    log_info("YOLO26 model loaded successfully");
    
    // Initialize tracker
    log_info("Initializing ByteTrack...");
    ByteTrackConfig config = bytetrack_default_config();
    ByteTracks* tracks = bytetrack_init(config);
    log_info("Tracker initialized");
    
    // Create video writer
    VideoWriter* writer = vw_open(output_path, width, height, 30.0f);
    if (!writer) {
        log_error("Failed to create output video");
    } else {
        log_info("Output video created: %s", output_path);
    }
    
    // Tracking history for visualization
    typedef struct {
        int track_id;
        float* points;  // Circular buffer of (x, y) points
        int capacity;
        int head;
        int count;
    } TrailHistory;
    
    TrailHistory* trails = NULL;
    int max_trails = 100;
    trails = (TrailHistory*)calloc(max_trails, sizeof(TrailHistory));
    
    // Process video frames
    int frame_count = 0;
    uint8_t** frame_buffer = &first_frame->data;  // Reuse first frame buffer
    
    while (g_running) {
        // Read frame (reuse buffer if possible)
        Image* frame = (frame_count == 0) ? first_frame : vc_read(cap);
        if (!frame) {
            break;
        }
        
        frame_count++;
        
        // Run YOLO detection
        YoloDetections* dets = yolo26_detect(yolo, frame->data, 
                                               frame->width, frame->height, 
                                               frame->channels);
        
        // Convert detections to flat array for tracker
        float* det_array = NULL;
        if (dets && dets->count > 0) {
            det_array = (float*)malloc(dets->count * 6 * sizeof(float));
            for (size_t i = 0; i < dets->count; i++) {
                det_array[i * 6 + 0] = dets->detections[i].x1;
                det_array[i * 6 + 1] = dets->detections[i].y1;
                det_array[i * 6 + 2] = dets->detections[i].x2;
                det_array[i * 6 + 3] = dets->detections[i].y2;
                det_array[i * 6 + 4] = dets->detections[i].confidence;
                det_array[i * 6 + 5] = (float)dets->detections[i].class_id;
            }
        }
        
        // Update tracker
        ByteTracks* tracked = bytetrack_update(tracks, det_array, 
                                                dets ? dets->count : 0,
                                                frame->data, 
                                                frame->width, frame->height);
        
        // Draw results
        if (tracked) {
            for (size_t i = 0; i < tracked->count; i++) {
                ByteTrack* t = &tracked->tracks[i];
                int x1 = (int)t->x1;
                int y1 = (int)t->y1;
                int x2 = (int)t->x2;
                int y2 = (int)t->y2;
                
                // Generate color based on track ID
                uint8_t r = ((t->track_id * 137) % 256);
                uint8_t g = ((t->track_id * 173) % 256);
                uint8_t b = ((t->track_id * 251) % 256);
                
                // Draw bounding box
                draw_box(frame->data, width, height, x1, y1, x2, y2, r, g, b, 2);
                
                // Draw label
                char label[64];
                snprintf(label, sizeof(label), "ID:%d %.2f", t->track_id, t->score);
                draw_text(frame->data, width, height, x1, y1 - 20, label, r, g, b);
                
                // Update trail
                int tid = t->track_id;
                if (tid >= 0 && tid < max_trails) {
                    TrailHistory* th = &trails[tid];
                    int cx = (x1 + x2) / 2;
                    int cy = (y1 + y2) / 2;
                    
                    if (th->capacity == 0) {
                        th->capacity = trail_length;
                        th->points = (float*)malloc(th->capacity * 2 * sizeof(float));
                        th->head = 0;
                        th->count = 0;
                    }
                    
                    th->points[th->head * 2] = (float)cx;
                    th->points[th->head * 2 + 1] = (float)cy;
                    th->head = (th->head + 1) % th->capacity;
                    if (th->count < th->capacity) th->count++;
                    
                    // Draw trail
                    for (int j = 1; j < th->count; j++) {
                        int idx = (th->head - j - 1 + th->capacity) % th->capacity;
                        int prev_idx = (th->head - j + th->capacity) % th->capacity;
                        float alpha = (float)j / th->count;
                        int thickness = 1 + (int)(alpha * 2);
                        
                        draw_line(frame->data, width, height,
                                 (int)th->points[prev_idx * 2],
                                 (int)th->points[prev_idx * 2 + 1],
                                 (int)th->points[idx * 2],
                                 (int)th->points[idx * 2 + 1],
                                 r, g, b, thickness);
                    }
                }
            }
        }
        
        // Draw frame info
        char info[128];
        snprintf(info, sizeof(info), "Frame: %d | Det: %zu | Track: %zu", 
                 frame_count, dets ? dets->count : 0, tracked ? tracked->count : 0);
        draw_text(frame->data, width, height, 10, 30, info, 0, 255, 0);
        
        // Write output video
        if (writer) {
            vw_write(writer, frame->data);
        }
        
        // Display window
        if (show_window) {
            // OpenCV imshow would be called here
            // cv::imshow("YOLO26 + ByteTrack", frame);
            // if (cv::waitKey(1) == 'q') break;
        }
        
        // Print progress
        if (frame_count % 30 == 0) {
            log_info("Processed %d frames, tracks: %zu", 
                     frame_count, tracked ? tracked->count : 0);
        }
        
        // Cleanup
        if (det_array) free(det_array);
        if (dets) yolo26_free_detections(dets);
        if (frame_count > 0) image_free(frame);
    }
    
    log_info("========================================");
    log_info("Processing complete!");
    log_info("Total frames: %d", frame_count);
    log_info("Output saved to: %s", output_path);
    log_info("========================================");
    
    // Cleanup
    if (trails) {
        for (int i = 0; i < max_trails; i++) {
            if (trails[i].points) free(trails[i].points);
        }
        free(trails);
    }
    
    if (writer) vw_close(writer);
    bytetrack_free(tracks);
    yolo26_destroy(yolo);
    vc_release(cap);
    
    return 0;
}
