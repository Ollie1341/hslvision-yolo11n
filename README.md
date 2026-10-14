# Yolo model comparison for Robo Cup (Booster K1)

## The current state of the NUbots K1 booster vision system

### Objective

The object of the curret model vision system for the Booster K1 was to trained, deploy and compete with the Booster K1 in Robo Cup : RCAP Beijing Masters 2026.

### Model

The current model that is deployed on the NUbots K1 is YOLO11n trained by [Tom0Brien](https://github.com/Tom0Brien/hslvision-yolo11n).

### Dataset

The current dataset is from [Team whIRLwind Amsterdam](https://huggingface.co/datasets/whirlwind-ams/hslvision) maded up for 6 classes with 6,068 images from different competiton venues, layouts and robot platforms.

#### Data collection

Images were collected across six locations spanning a range of field layouts, goalpost designs, ball sizes, robot platforms, and HSL field sizes (small, middle, large):

- LAB42: whIRLwind
- German Open 2026, Cologne: whIRLwind
- RCAP Beijing Masters 2025: whIRLwind, HTWK
- RoboCup 2025, Salvador: HTWK
- RoboCup 2019, Sydney: BitBots
- RoboCup 2017, Nagoya: BitBots

### Classes

The is current 6 categories that we are trying to they include the follwing :

1. Ball
2. Robot
3. Goalpost
4. Penalty marks
5. X-Intersections
6. Y-Intersections

### Reasoning to change models

The reason to change models is because YOLO26 is better performancing which well be shown in the ladder part of the report.

## Accuracy and Performance Metrics

| Model   | Batch | Epoch Max | Epoch Reached | Patiance |
| :------ | :---: | :-------: | :-----------: | :------: |
| YOLO11n |  16   |   Right   |      200      |    25    |
| YOLO26n |  32   |   Text    |      500      |    25    |
| YOLO26s |  32   |   Text    |      500      |    25    |
| YOLO26m |  222  |   Text    |      500      |    25    |

## Speed and lantency

## Architecture and Model Characteristics

### Mean Average Precision (mAP)

### Mean Average Precision between 50-95 (mAP50-95)

### Mean Average Precision 70 (P)

### Mean Average Recall (R)

## Deployment and Operational Factors

### Hardware Compatibility

NUbots team currently uses the Booster K1 : Education which uses an NVIDIA Jetson Orin NX (8 GB) module delivering up to 117 TOPS. The current camera is a stero RGB module that runs at 25fps. These are important to know as they directly affect the speed of the model and how fast the model has to be to keep up the the hardware.

### Ecosystem and Tooling

The current eco system is

### Licensing
