简体中文 | [English](./README_EN.md)

# axcl-samples

## Introduction

**AXCL-Samples** is developed by [AXERA](https://www.axera-tech.com/). This project provides sample code for running common **open-source deep learning algorithms** on **PCIE accelerator cards** powered by **AXERA** SoCs, enabling community developers to quickly evaluate and adapt.

### Supported OS

- Ubuntu
- Debian
- Windows 11

### Supported Boards

| Board | Image | Chip | Vendor | Link |
| ----- | ----- | ---- | ------ | ---- |
| AI Core AX-M1 | <img src="docs/boards/aicore_ax_m1_top.jpg" width="200"> | AX650N | Radxa | [Docs](https://docs.radxa.com/en/aicore/ax-m1) |
| M4Chat | <img src="docs/boards/m4chat.jpg" width="200"> | AX8850 | Sipeed | [Wiki](https://wiki.sipeed.com/hardware/zh/maixIV/m4chat/intro.html) |
| LLM-8850 Card | <img src="docs/boards/AI-001_LLM-8850-main-pictures_01.jpg" width="200"> | AX8850 | M5Stack | [Docs](https://docs.m5stack.com/zh_CN/ai_hardware/LLM-8850_Card) |

## AXCL

**AXCL** is a C/Python API library for developing deep neural network inference, transcoding and other applications on AXERA chip platforms. It provides APIs for runtime resource management, memory management, model loading and execution, and media data processing.

- [Online Documentation](https://axcl-docs.readthedocs.io/en/latest/index.html)

## Quick Start

### Build from Source

- Ensure AXCL deb packages are properly installed following the [AXCL Documentation](https://axcl-docs.readthedocs.io/en/latest/index.html). Header files and libraries should be installed at `/usr/include/axcl/` and `/usr/lib/axcl/` respectively.
- The following example is demonstrated on Raspberry Pi 5.

#### Clone the Repository

```
git clone https://github.com/AXERA-TECH/axcl-samples.git
```

#### Install Build Tools

Install the required build tools via `apt install`:

```
sudo apt update
sudo apt install build-essential cmake libopencv-dev 
```

#### Build

```
mkdir build && cd build
cmake ..
make install -j4
```

After building, the sample binaries will be generated under `./install/bin`:

```
axera@raspberrypi:~/temp/axcl-samples/build $ tree install
install
└── bin
    ├── ax_classification
    ├── ax_depth_anything
    ├── ax_yolo11
    ├── ax_yolo11_pose
    ├── ax_yolo11_seg
    ├── ax_yolov10
    ├── ax_yolov10_u
    ├── ax_yolov5_face
    ├── ax_yolov5s
    ├── ax_yolov5s_seg
    ├── ax_yolov8
    ├── ax_yolov8_pose
    ├── ax_yolov8_seg
    ├── ax_yolov9
    └── ax_yolov9_u
```

## Run Examples

```
axera@raspberrypi:~/temp/axcl-samples/build $ ./install/bin/ax_yolo11 -m yolo11x.axmodel -i ssd_horse.jpg
--------------------------------------
model file : yolo11x.axmodel
image file : ssd_horse.jpg
img_h, img_w : 640 640
--------------------------------------

input size: 1
    name:   images [unknown] [unknown]
        1 x 640 x 640 x 3


output size: 3
    name: /model.23/Concat_output_0
        1 x 80 x 80 x 144

    name: /model.23/Concat_1_output_0
        1 x 40 x 40 x 144

    name: /model.23/Concat_2_output_0
        1 x 20 x 20 x 144

==================================================

Engine push input is done.
--------------------------------------
post process cost time:1.09 ms
--------------------------------------
Repeat 1 times, avg time 43.09 ms, max_time 43.09 ms, min_time 43.09 ms
--------------------------------------
detection num: 6
17:  96%, [ 216,   71,  423,  370], horse
16:  93%, [ 144,  203,  196,  345], dog
 0:  89%, [ 273,   14,  349,  231], person
 2:  88%, [   1,  105,  132,  197], car
 0:  82%, [ 431,  124,  451,  178], person
19:  46%, [ 171,  137,  202,  169], cow
--------------------------------------
```

## Resources

### Model Resources

- **ModelZoo**, **pre-built binaries**, **test images** and more:
  - [Huggingface](https://huggingface.co/collections/AXERA-TECH/vision-models-67b0bce92ddc61229e8e94ed)
  - [Modelscope](https://modelscope.cn/organization/AXERA-TECH)

### NPU Toolchain

NPU toolchain documentation and downloads:
  - [Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/) (Support AX650A/AX650N/AX630C/AX620Q)

## Community

- Github issues
- QQ Group: 139953715
