

# A new efficient lightweight semantic segmentation Network(E-UNet)

## 1. Introduction

As explained in the previous post, semantic segmentation can be described as classifying each pixel into a specific class which can be employed for many goals such as satellite imagery analysis to on-the-fly visual search, preservation of the cultural heritage up to the recognition of image copies and human-computer interaction.

Different deep learning models have been introduced to segment objects semantically, but the ability and need to perform pixel-by-pixel semantic segmentation in real time and with fewer FLOPs (floating point operations per second) to achieve similar or better accuracy than existing models such as U-net is one of the major challenges that must be considered.

To overcome this challenge I tried to improve the ENet (A Deep Neural Network Architecture for Real-Time Semantic Segmentation) by removing some layers and adding skip connection to have a lightweight semantic segmentation network called E-UNet which looks like a “U
