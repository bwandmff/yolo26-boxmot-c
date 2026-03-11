/*
 * bytetrack.h - ByteTrack multi-object tracking header
 */

#ifndef BYTETRACK_H
#define BYTETRACK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Track structure
typedef struct {
    int track_id;
    float x1, y1, x2, y2;   // Current bounding box
    float score;             // Detection score
    int class_id;
    int frames_since_seen;
    int age;                 // Frames since first appearance
    float* history_x;        // Center position history for velocity estimation
    float* history_y;
    int history_size;
} ByteTrack;

// Track array
typedef struct {
    ByteTrack* tracks;
    size_t count;
    size_t capacity;
} ByteTracks;

// Tracking parameters
typedef struct {
    float track_thresh;      // Threshold for new track
    float track_buffer;     // Frames to keep lost tracks
    float match_thresh;    // IOU threshold for matching
    float min_box_area;    // Minimum box area
    int max_time_lost;     // Max frames lost before removing
} ByteTrackConfig;

// Default configuration
ByteTrackConfig bytetrack_default_config(void);

// Initialize ByteTrack
ByteTracks* bytetrack_init(ByteTrackConfig config);

// Update tracks with new detections
// detections should be in format: [x1, y1, x2, y2, score, class_id, ...]
ByteTracks* bytetrack_update(ByteTracks* tracker, const float* detections, 
                             size_t det_count, float* image_data, 
                             int img_width, int img_height);

// Free tracks
void bytetrack_free(ByteTracks* tracks);

#ifdef __cplusplus
}
#endif

#endif // BYTETRACK_H
