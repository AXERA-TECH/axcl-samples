# AXERA is pleased to support the open source community by making ax-samples available.
#
# Copyright (c) 2022, AXERA Semiconductor (Shanghai) Co., Ltd. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# Author:
#

find_package(Threads)
find_package(OpenCV REQUIRED)

include("${CMAKE_SOURCE_DIR}/cmake/axcl.cmake")

axera_example(axcl_classification ax_classification_steps.cc)

axera_example(axcl_yolov5s ax_yolov5s_steps.cc)
axera_example(axcl_yolov5s_seg ax_yolov5s_seg_steps.cc)
axera_example(axcl_yolov5_face ax_yolov5_face_steps.cc)
axera_example(axcl_yolov7_face ax_yolov7_face_steps.cc)
axera_example(axcl_yolov8 ax_yolov8_steps.cc)
axera_example(axcl_yolov8_seg ax_yolov8_seg_steps.cc)
axera_example(axcl_yolov8_pose ax_yolov8_pose_steps.cc)
axera_example(axcl_yolov9 ax_yolov9_steps.cc)
axera_example(axcl_yolov9_u ax_yolov9_ultralytics_steps.cc)
axera_example(axcl_yolov10 ax_yolov10s_steps.cc)
axera_example(axcl_yolov10_u ax_yolov10_ultralytics_steps.cc)
axera_example(axcl_yolo11 ax_yolo11_steps.cc)
axera_example(axcl_yolo11_seg ax_yolo11_seg_steps.cc)
axera_example(axcl_yolo11_pose ax_yolo11_pose_steps.cc)
axera_example(axcl_yolo26 ax_yolo26_steps.cc)
axera_example(axcl_yolo_world_open_vocabulary ax_yolo_world_open_vocabulary_steps.cc)
axera_example(axcl_palm_detection ax_palm_detection_steps.cc)
axera_example(axcl_handpose ax_handpose_steps.cc)
axera_example(axcl_depth_anything ax_depth_anything_steps.cc)

