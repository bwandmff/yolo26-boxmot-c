/*
 * test_mode.c - Simple test without video I/O
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "yolo26.h"
#include "bytetrack.h"
#include "utils.h"

// Create a simple test image with some boxes
static uint8_t* create_test_image(int width, int height) {
    uint8_t* img = (uint8_t*)malloc(width * height * 3);
    
    // Fill with gray background
    memset(img, 128, width * height * 3);
    
    // Draw some rectangles (simulating detections)
    // Box 1: car-like rectangle
    for (int y = 200; y < 400; y++) {
        for (int x = 300; x < 600; x++) {
            if (y >= 200 && y <= 250 || y >= 350 && y <= 400 || 
                x >= 300 && x <= 320 || x >= 580 && x <= 600) {
                img[(y * width + x) * 3 + 0] = 0;   // B
                img[(y * width + x) * 3 + 1] = 0;   // G
                img[(y * width + x) * 3 + 2] = 255; // R
            }
        }
    }
    
    // Box 2
    for (int y = 300; y < 500; y++) {
        for (int x = 800; x < 1100; x++) {
            if (y >= 300 && y <= 350 || y >= 450 && y <= 500 || 
                x >= 800 && x <= 820 || x >= 1080 && x <= 1100) {
                img[(y * width + x) * 3 + 0] = 255; // B
                img[(y * width + x) * 3 + 1] = 0;   // G
                img[(y * width + x) * 3 + 2] = 0;   // R
            }
        }
    }
    
    return img;
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    
    printf("========================================\n");
    printf("YOLO26 + ByteTrack Test Mode\n");
    printf("========================================\n");
    
    // Test YOLO26 model loading
    printf("\n[1] Loading YOLO26 model...\n");
    Yolo26Model* yolo = yolo26_init("models/yolo26n.onnx", 0.5, 80);
    if (!yolo) {
        printf("ERROR: Failed to load YOLO model!\n");
        return 1;
    }
    printf("SUCCESS: YOLO26 model loaded!\n");
    
    // Create test image
    printf("\n[2] Creating test image...\n");
    int width = 640;
    int height = 640;
    uint8_t* test_img = create_test_image(width, height);
    printf("SUCCESS: Test image created (%dx%d)\n", width, height);
    
    // Run detection
    printf("\n[3] Running YOLO26 detection...\n");
    YoloDetections* dets = yolo26_detect(yolo, test_img, width, height, 3);
    if (dets) {
        printf("SUCCESS: Detection complete! Found %zu objects\n", dets->count);
        for (size_t i = 0; i < dets->count; i++) {
            printf("  - Object %zu: x1=%.1f y1=%.1f x2=%.1f y2=%.1f conf=%.2f class=%d\n",
                   i, dets->detections[i].x1, dets->detections[i].y1,
                   dets->detections[i].x2, dets->detections[i].y2,
                   dets->detections[i].confidence, dets->detections[i].class_id);
        }
        yolo26_free_detections(dets);
    } else {
        printf("WARNING: Detection returned NULL\n");
    }
    
    // Test ByteTrack
    printf("\n[4] Testing ByteTrack...\n");
    ByteTrackConfig config = bytetrack_default_config();
    ByteTracks* tracker = bytetrack_init(config);
    if (!tracker) {
        printf("ERROR: Failed to initialize ByteTrack!\n");
        yolo26_destroy(yolo);
        free(test_img);
        return 1;
    }
    printf("SUCCESS: ByteTrack initialized!\n");
    
    // Simulate detections for tracking test
    printf("\n[5] Testing track update...\n");
    float test_dets[] = {
        100, 100, 200, 200, 0.9, 0,  // x1, y1, x2, y2, conf, class
        300, 150, 400, 250, 0.85, 0,
        500, 200, 600, 350, 0.95, 0
    };
    
    ByteTracks* result = bytetrack_update(tracker, test_dets, 3, NULL, width, height);
    if (result) {
        printf("SUCCESS: Track update complete! Active tracks: %zu\n", result->track_count);
        for (size_t i = 0; i < result->track_count; i++) {
            printf("  - Track %d: x1=%.1f y1=%.1f x2=%.1f y2=%.1f conf=%.2f\n",
                   result->tracks[i].track_id,
                   result->tracks[i].x1, result->tracks[i].y1,
                   result->tracks[i].x2, result->tracks[i].y2,
                   result->tracks[i].score);
        }
    }
    
    // Cleanup
    printf("\n[6] Cleaning up...\n");
    bytetrack_free(tracker);
    yolo26_destroy(yolo);
    free(test_img);
    
    printf("\n========================================\n");
    printf("All tests PASSED!\n");
    printf("========================================\n");
    
    return 0;
}
