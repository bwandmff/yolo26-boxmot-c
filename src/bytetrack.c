/*
 * bytetrack.c - ByteTrack multi-object tracking implementation
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "bytetrack.h"

// Maximum number of tracks
#define MAX_TRACKS 256
#define MAX_HISTORY 10

struct ByteTrack {
    ByteTrackConfig config;
    ByteTrack* tracks;       // Active tracks
    size_t track_count;
    ByteTrack* lost_tracks;  // Lost tracks (for recovery)
    size_t lost_count;
    int next_track_id;
};

static float calculate_iou(float x1, float y1, float x2, float y2,
                           float x1_, float y1_, float x2_, float y2_) {
    // Calculate intersection
    float inter_x1 = fmaxf(x1, x1_);
    float inter_y1 = fmaxf(y1, y1_);
    float inter_x2 = fminf(x2, x2_);
    float inter_y2 = fminf(y2, y2_);
    
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

static float calculate_distance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

ByteTrackConfig bytetrack_default_config(void) {
    ByteTrackConfig config;
    config.track_thresh = 0.5f;
    config.track_buffer = 30;
    config.match_thresh = 0.3f;
    config.min_box_area = 10.0f;
    config.max_time_lost = 30;
    return config;
}

ByteTracks* bytetrack_init(ByteTrackConfig config) {
    ByteTracks* tracker = (ByteTracks*)malloc(sizeof(ByteTracks));
    if (!tracker) {
        return NULL;
    }
    
    tracker->tracks = (ByteTrack*)malloc(sizeof(ByteTrack) * MAX_TRACKS);
    tracker->capacity = MAX_TRACKS;
    tracker->count = 0;
    
    // Lost tracks buffer
    tracker->lost_tracks = (ByteTrack*)malloc(sizeof(ByteTrack) * MAX_TRACKS);
    tracker->lost_count = 0;
    
    return tracker;
}

static void remove_track(ByteTracks* tracker, int index) {
    if (index < 0 || index >= (int)tracker->count) {
        return;
    }
    
    // Free history
    if (tracker->tracks[index].history_x) {
        free(tracker->tracks[index].history_x);
    }
    if (tracker->tracks[index].history_y) {
        free(tracker->tracks[index].history_y);
    }
    
    // Shift tracks
    for (size_t i = index; i < tracker->count - 1; i++) {
        tracker->tracks[i] = tracker->tracks[i + 1];
    }
    tracker->count--;
}

ByteTracks* bytetrack_update(ByteTracks* tracker, const float* detections,
                             size_t det_count, float* image_data,
                             int img_width, int img_height) {
    if (!tracker) {
        return NULL;
    }
    
    // Separate detections into high and low score
    float* high_detections = NULL;
    float* low_detections = NULL;
    size_t high_count = 0;
    size_t low_count = 0;
    
    if (detections && det_count > 0) {
        for (size_t i = 0; i < det_count; i++) {
            float score = detections[i * 6 + 4];
            if (score >= tracker->config.track_thresh) {
                high_count++;
            } else {
                low_count++;
            }
        }
        
        if (high_count > 0) {
            high_detections = (float*)malloc(high_count * 6 * sizeof(float));
        }
        if (low_count > 0) {
            low_detections = (float*)malloc(low_count * 6 * sizeof(float));
        }
        
        size_t hi = 0, li = 0;
        for (size_t i = 0; i < det_count; i++) {
            float score = detections[i * 6 + 4];
            if (score >= tracker->config.track_thresh) {
                memcpy(&high_detections[hi * 6], &detections[i * 6], 6 * sizeof(float));
                hi++;
            } else {
                memcpy(&low_detections[li * 6], &detections[i * 6], 6 * sizeof(float));
                li++;
            }
        }
    }
    
    // Step 1: Predict new locations using Kalman filter (simplified linear motion)
    for (size_t i = 0; i < tracker->count; i++) {
        ByteTrack* track = &tracker->tracks[i];
        
        // Simple linear prediction based on velocity
        if (track->history_size > 1) {
            float vx = track->history_x[track->history_size - 1] - 
                      track->history_x[track->history_size - 2];
            float vy = track->history_y[track->history_size - 1] - 
                      track->history_y[track->history_size - 2];
            
            track->x1 += vx;
            track->y1 += vy;
            track->x2 += vx;
            track->y2 += vy;
        }
        
        track->frames_since_seen++;
        track->age++;
        
        // Update center for history
        float cx = (track->x1 + track->x2) / 2.0f;
        float cy = (track->y1 + track->y2) / 2.0f;
        
        // Update history
        if (track->history_size < MAX_HISTORY) {
            if (!track->history_x) {
                track->history_x = (float*)malloc(MAX_HISTORY * sizeof(float));
                track->history_y = (float*)malloc(MAX_HISTORY * sizeof(float));
            }
            track->history_x[track->history_size] = cx;
            track->history_y[track->history_size] = cy;
            track->history_size++;
        }
    }
    
    // Step 2: Associate high score detections with active tracks
    if (high_detections && high_count > 0 && tracker->count > 0) {
        // Create cost matrix
        float cost_matrix[MAX_TRACKS][128];  // Max 128 detections
        int match_matrix[MAX_TRACKS][128];
        
        // Initialize matches
        for (size_t i = 0; i < tracker->count; i++) {
            for (size_t j = 0; j < high_count; j++) {
                cost_matrix[i][j] = 1.0f;  // 1 = no match
                match_matrix[i][j] = 0;
            }
        }
        
        // Calculate IOU costs
        for (size_t i = 0; i < tracker->count; i++) {
            for (size_t j = 0; j < high_count; j++) {
                float iou = calculate_iou(
                    tracker->tracks[i].x1, tracker->tracks[i].y1,
                    tracker->tracks[i].x2, tracker->tracks[i].y2,
                    high_detections[j * 6 + 0], high_detections[j * 6 + 1],
                    high_detections[j * 6 + 2], high_detections[j * 6 + 3]
                );
                cost_matrix[i][j] = 1.0f - iou;  // Lower is better
                if (iou > tracker->config.match_thresh) {
                    match_matrix[i][j] = 1;
                }
            }
        }
        
        // Greedy matching
        int matched_dets[128] = {0};
        for (size_t i = 0; i < tracker->count; i++) {
            for (size_t j = 0; j < high_count; j++) {
                if (!matched_dets[j] && match_matrix[i][j]) {
                    // Update track
                    tracker->tracks[i].x1 = high_detections[j * 6 + 0];
                    tracker->tracks[i].y1 = high_detections[j * 6 + 1];
                    tracker->tracks[i].x2 = high_detections[j * 6 + 2];
                    tracker->tracks[i].y2 = high_detections[j * 6 + 3];
                    tracker->tracks[i].score = high_detections[j * 6 + 4];
                    tracker->tracks[i].class_id = (int)high_detections[j * 6 + 5];
                    tracker->tracks[i].frames_since_seen = 0;
                    
                    matched_dets[j] = 1;
                    break;
                }
            }
        }
        
        // Mark unmatched tracks
        for (size_t i = 0; i < tracker->count; i++) {
            int matched = 0;
            for (size_t j = 0; j < high_count; j++) {
                if (matched_dets[j] && match_matrix[i][j]) {
                    matched = 1;
                    break;
                }
            }
            if (!matched) {
                // Move to lost tracks
                tracker->tracks[i].frames_since_seen = 0;
            }
        }
    }
    
    // Step 3: Create new tracks from unmatched high score detections
    if (high_detections && high_count > 0) {
        int matched[128] = {0};
        
        // Check against existing tracks
        for (size_t i = 0; i < tracker->count; i++) {
            for (size_t j = 0; j < high_count; j++) {
                float iou = calculate_iou(
                    tracker->tracks[i].x1, tracker->tracks[i].y1,
                    tracker->tracks[i].x2, tracker->tracks[i].y2,
                    high_detections[j * 6 + 0], high_detections[j * 6 + 1],
                    high_detections[j * 6 + 2], high_detections[j * 6 + 3]
                );
                if (iou > tracker->config.match_thresh) {
                    matched[j] = 1;
                }
            }
        }
        
        // Create new tracks
        for (size_t j = 0; j < high_count; j++) {
            if (!matched[j] && tracker->count < MAX_TRACKS - 1) {
                ByteTrack* new_track = &tracker->tracks[tracker->count];
                new_track->track_id = tracker->next_track_id++;
                new_track->x1 = high_detections[j * 6 + 0];
                new_track->y1 = high_detections[j * 6 + 1];
                new_track->x2 = high_detections[j * 6 + 2];
                new_track->y2 = high_detections[j * 6 + 3];
                new_track->score = high_detections[j * 6 + 4];
                new_track->class_id = (int)high_detections[j * 6 + 5];
                new_track->frames_since_seen = 0;
                new_track->age = 0;
                new_track->history_x = NULL;
                new_track->history_y = NULL;
                new_track->history_size = 0;
                
                tracker->count++;
            }
        }
    }
    
    // Step 4: Remove old tracks
    for (int i = (int)tracker->count - 1; i >= 0; i--) {
        if (tracker->tracks[i].frames_since_seen > tracker->config.max_time_lost) {
            remove_track(tracker, i);
        }
    }
    
    // Cleanup
    if (high_detections) free(high_detections);
    if (low_detections) free(low_detections);
    
    return tracker;
}

void bytetrack_free(ByteTracks* tracks) {
    if (!tracks) {
        return;
    }
    
    if (tracks->tracks) {
        for (size_t i = 0; i < tracks->count; i++) {
            if (tracks->tracks[i].history_x) free(tracks->tracks[i].history_x);
            if (tracks->tracks[i].history_y) free(tracks->tracks[i].history_y);
        }
        free(tracks->tracks);
    }
    
    if (tracks->lost_tracks) {
        for (size_t i = 0; i < tracks->lost_count; i++) {
            if (tracks->lost_tracks[i].history_x) free(tracks->lost_tracks[i].history_x);
            if (tracks->lost_tracks[i].history_y) free(tracks->lost_tracks[i].history_y);
        }
        free(tracks->lost_tracks);
    }
    
    free(tracks);
}
