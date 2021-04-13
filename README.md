# PMED-Net: Pyramid Based Multi-Scale Encoder-Decoder Network for Medical Image Segmentation
This repository contains implementation details, pre-trained models and datasets information  of "PMED-Net: Pyramid Based Multi-Scale Encoder-Decoder Network for Medical Image Segmentation".
All the models are trained and tested using Keras framework with Tensorflow backend on a a PC equipped with an NVIDIA Titan XP GPU.

# Overview of PMED-Net

The overall PMED-Net architecture is shown in below Figure. The proposed model consists of six small encoder-decoder networks, where each generates coarse predictions, which are further refined by the next level network.  Predictions of the previous model are concatenated with the input image at different scales. The proposed cascaded methodology enables the network to re-use the information iteratively and extract the features at different resolutions.

![image](https://user-images.githubusercontent.com/56618776/114500024-40092000-9c62-11eb-9927-7408917fa743.png)


A three-stage encoder-decoder network is trained independently to estimate the segmentation map. The detailed architecture of a single light-weighted encoder-decoder network is shown in below figure.

![image](https://user-images.githubusercontent.com/56618776/114500070-56af7700-9c62-11eb-8cb4-498b9f261c7b.png)


 # Experimental outputs of the ISIC dataset for different networks. 
![image](https://user-images.githubusercontent.com/56618776/114500339-c4f43980-9c62-11eb-840a-1006a00eb4e8.png)
