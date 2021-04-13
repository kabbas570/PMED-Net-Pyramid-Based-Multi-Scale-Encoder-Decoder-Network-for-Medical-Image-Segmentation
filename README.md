# PMED-Net: Pyramid Based Multi-Scale Encoder-Decoder Network for Medical Image Segmentation
This repository contains implementation details, pre-trained models and datasets information  of "PMED-Net: Pyramid Based Multi-Scale Encoder-Decoder Network for Medical Image Segmentation".
All the models are trained and tested using Keras framework with Tensorflow backend on a a PC equipped with an NVIDIA Titan XP GPU.
 The paper has been accepted by IEEE Acess jounral and can be accessed at:
 https://ieeexplore.ieee.org/document/9399071


# Overview of PMED-Net

The overall PMED-Net architecture is shown in below Figure. The proposed model consists of six small encoder-decoder networks, where each generates coarse predictions, which are further refined by the next level network.  Predictions of the previous model are concatenated with the input image at different scales. The proposed cascaded methodology enables the network to re-use the information iteratively and extract the features at different resolutions.

![image](https://user-images.githubusercontent.com/56618776/114500024-40092000-9c62-11eb-9927-7408917fa743.png)


A three-stage encoder-decoder network is trained independently to estimate the segmentation map. The detailed architecture of a single light-weighted encoder-decoder network is shown in below figure.

![image](https://user-images.githubusercontent.com/56618776/114500070-56af7700-9c62-11eb-8cb4-498b9f261c7b.png)


 ## Experimental outputs of the ISIC dataset for different networks
![image](https://user-images.githubusercontent.com/56618776/114500466-04228a80-9c63-11eb-9924-64aa2dd7de04.png)

##  Experimental results of different networks for the Nuclei dataset
![image](https://user-images.githubusercontent.com/56618776/114500580-43e97200-9c63-11eb-989c-6d94ddb7e756.png)
 ## Qualitative comparison of results of segmentation for brain tumor dataset
 ![image](https://user-images.githubusercontent.com/56618776/114500654-6e3b2f80-9c63-11eb-855d-dbd5809a0d0e.png)
## Visual results for NIH X-ray dataset segmentation
![image](https://user-images.githubusercontent.com/56618776/114500709-93c83900-9c63-11eb-9a5d-84f5c5e2cca1.png)
