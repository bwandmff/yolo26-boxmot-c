# Makefile for YOLO26 + ByteTrack Linux C Implementation

# Compiler
CC = gcc
CXX = g++

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = .

# OpenCV (optional)
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)

# ONNX Runtime
ONNX_CFLAGS = -I/usr/local/include
ONNX_LIBS = -L/usr/local/lib -lonnxruntime

# If ONNX Runtime not installed, use system paths
ONNX_CFLAGS_SYSTEM = $(shell ls -d /usr/include/onnxruntime* 2>/dev/null | head -1)
ifneq ($(ONNX_CFLAGS_SYSTEM),)
    ONNX_CFLAGS = -I$(ONNX_CFLAGS_SYSTEM)
    ONNX_LIBS = $(shell ls /usr/lib/libonnxruntime* 2>/dev/null | head -1 | sed 's/^/-l/')
endif

# Flags
CFLAGS = -Wall -Wextra -O3 -std=c11 -I$(INC_DIR) -DUSE_OPENCV $(OPENCV_CFLAGS) $(ONNX_CFLAGS)
CXXFLAGS = -Wall -Wextra -O3 -std=c++17 -I$(INC_DIR) $(OPENCV_CFLAGS) $(ONNX_CFLAGS)

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Dependencies
DEPS = $(wildcard $(INC_DIR)/*.h)

# Target
TARGET = yolo26_bytetrack

# Default target
.PHONY: all
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Link
$(TARGET): $(OBJECTS)
	$(CC) $^ -o $@ $(OPENCV_LIBS) $(ONNX_LIBS) -lm -lpthread -ldl

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(DEPS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile main
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Add main.o to OBJECTS
OBJECTS += $(BUILD_DIR)/main.o

# Debug build
.PHONY: debug
debug: CFLAGS += -g -DDEBUG
debug: clean all

# Clean
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Install dependencies (Ubuntu/Debian)
.PHONY: deps
deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y \
		build-essential \
		libopencv-dev \
		libonnxruntime-dev

# Download ONNX model
.PHONY: get-model
get-model:
	@echo "Downloading YOLO26n ONNX model..."
	mkdir -p models
	curl -L -o models/yolo26n.onnx \
		"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.onnx"

# Run
.PHONY: run
run: all get-model
	./$(TARGET) --model models/yolo26n.onnx --source 0

# Run with video file
.PHONY: run-video
run-video: all
	./$(TARGET) --model models/yolo26n.onnx --source video.mp4 --output output.mp4

# Help
.PHONY: help
help:
	@echo "YOLO26 + ByteTrack Linux C Implementation"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build the project"
	@echo "  clean       - Remove build files"
	@echo "  deps        - Install system dependencies (Ubuntu)"
	@echo "  get-model   - Download YOLO26n ONNX model"
	@echo "  run         - Run with webcam"
	@echo "  run-video   - Run with video file"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  make deps"
	@echo "  make get-model"
	@echo "  make run-video SOURCE=video.mp4"

# Show configuration
.PHONY: info
info:
	@echo "OpenCV flags: $(OPENCV_CFLAGS)"
	@echo "OpenCV libs: $(OPENCV_LIBS)"
	@echo "ONNX flags: $(ONNX_CFLAGS)"
	@echo "ONNX libs: $(ONNX_LIBS)"
