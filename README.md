[English](./README_EN.md) | 简体中文

# axcl-samples

## 简介

**AXCL-Samples** 由 爱芯元智 主导开发。该项目实现了常见的**深度学习开源算法**在基于 **爱芯元智** 的 SoC 实现的 **PCIE算力卡** 产品上的运行的示例代码，方便社区开发者进行快速评估和适配。

### 支持系统

- Ubuntu
- Debian
- Windows 11

### 支持板卡

| 板卡 | 图片 | 芯片 | 厂商 | 链接 |
| ---- | ---- | ---- | ---- | ---- |
| AI Core AX-M1 | <img src="docs/boards/aicore_ax_m1_top.jpg" width="200"> | AX650N | Radxa | [文档](https://docs.radxa.com/en/aicore/ax-m1) |
| M4Chat | <img src="docs/boards/m4chat.jpg" width="200"> | AX8850 | Sipeed | [Wiki](https://wiki.sipeed.com/hardware/zh/maixIV/m4chat/intro.html) |
| LLM-8850 Card | <img src="docs/boards/AI-001_LLM-8850-main-pictures_01.jpg" width="200"> | AX8850 | M5Stack | [文档](https://docs.m5stack.com/zh_CN/ai_hardware/LLM-8850_Card) |

## AXCL

**AXCL** 是用于在 Axera 芯片平台上开发深度神经网络推理、转码等应用的 C、Python 语言 API 库，提供运行资源管理，内存管理，模型加载和执行，媒体数据处理等 API。

- [在线文档](https://axcl-docs.readthedocs.io/zh-cn/latest/index.html)

## 快速上手

### 本地编译

- 默认已经按照 [AXCL在线文档](https://axcl-docs.readthedocs.io/zh-cn/latest/index.html) 说明正确完成 AXCL 相关 deb 包安装，相关头文件和库文件分别已安装在 `/usr/include/axcl/` 和 `/usr/lib/axcl/` 路径下；
- 本示例在 Raspberry Pi 5 上进行操作。

#### 下载项目

```
git clone https://github.com/AXERA-TECH/axcl-samples.git
```

#### 安装编译工具
通过 `apt install` 安装必要的编译工具

```
sudo apt update
sudo apt install build-essential cmake libopencv-dev 
```

#### 编译详情

```
mkdir build && cd build
cmake ..
make install -j4
```

编译完成后在 `./install/bin` 下生成相关示例程序

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

## 示例运行

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

## 其他资源

### 网盘资源

- 提供 **ModelZoo**, **预编译程序**, **测试图片** 等内容:
  - [Huggingface](https://huggingface.co/collections/AXERA-TECH/vision-models-67b0bce92ddc61229e8e94ed)
  - [Modelscope](https://modelscope.cn/organization/AXERA-TECH)

### NPU 工具链

提供了NPU工具链相关使用说明和获取方式
  - [Pulsar2](https://pulsar2-docs.readthedocs.io/zh_CN/latest/)(Support AX650A/AX650N/AX630C/AX620Q)

## 技术讨论

- Github issues
- QQ 群: 139953715
