# PMED-Net: Pyramid Based Multi-Scale Encoder-Decoder Network for Medical Image Segmentation
This repository contains implementation details, pre-trained models and datasets information  of "PMED-Net: Pyramid Based Multi-Scale Encoder-Decoder Network for Medical Image Segmentation".
All the models are trained and tested using Keras framework with Tensorflow backend on a a PC equipped with an NVIDIA Titan XP GPU.
 ### The paper has been accepted by IEEE Access jounral and available at:
 https://ieeexplore.ieee.org/document/9399071


# Overview of PMED-Net

The overall PMED-Net architecture is shown in below Figure. The proposed model consists of six small encoder-decoder networks, where each generates coarse predictions, which are further refined by the next level network.  Predictions of the previous model are concatenated with the input image at different scales. The proposed cascaded methodology enables the network to re-use the information iteratively and extract the features at different resolutions.

![image](https://user-images.githubusercontent.com/56618776/114500024-40092000-9c62-11eb-9927-7408917fa743.png)


A three-stage encoder-decoder network is trained independently to estimate the segmentation map. The detailed architecture of a single light-weighted encoder-decoder network is shown in below figure.

![image](https://user-images.githubusercontent.com/56618776/114500070-56af7700-9c62-11eb-8cb4-498b9f261c7b.png)

# Ablation Studies and Model Size comparisons

•	In terms of model parameters, the proposed architecture is 95.30% smaller than SegNet [25], 95.27% than U-Net [26], 92.90% than BCDU-Net [27], 91.42% than CU-Net [28], 91.11% smaller than FCN-8s [29], 84.94% than ORED-Net [30], and 79.81% smaller than MultiResUNet [31].

IoU plotted as a function of number of pyramid level or number of encoder-decoder networks〖 N〗_k  , set in cascade for each dataset.

![image](https://user-images.githubusercontent.com/56618776/114501769-99bf1980-9c65-11eb-9ffb-e11285771185.png)


 ## Experimental outputs of the ISIC dataset for different networks
![image](https://user-images.githubusercontent.com/56618776/114500466-04228a80-9c63-11eb-9924-64aa2dd7de04.png)

##  Experimental results of different networks for the Nuclei dataset
![image](https://user-images.githubusercontent.com/56618776/114500580-43e97200-9c63-11eb-989c-6d94ddb7e756.png)
 ## Qualitative comparison of results of segmentation for brain tumor dataset
 ![image](https://user-images.githubusercontent.com/56618776/114500654-6e3b2f80-9c63-11eb-855d-dbd5809a0d0e.png)
## Visual results for NIH X-ray dataset segmentation
![image](https://user-images.githubusercontent.com/56618776/114500709-93c83900-9c63-11eb-9a5d-84f5c5e2cca1.png)

# Citation Request
If you use CED-Net in your project, please cite the following paper

A. Khan, H. Kim and L. Chua, "PMED-Net: Pyramid Based Multi-Scale Encoder-Decoder Network for Medical Image Segmentation," in IEEE Access, vol. 9, pp. 55988-55998, 2021, doi: 10.1109/ACCESS.2021.3071754.


  author={A. {Khan} and H. {Kim} and L. {Chua}},
  
  journal={IEEE Access}, 
  
  title={PMED-Net: Pyramid Based Multi-Scale Encoder-Decoder Network for Medical Image Segmentation}, 
  
  year={2021},
  
  volume={9},
  
  number={},
  
  pages={55988-55998},
  
  doi={10.1109/ACCESS.2021.3071754}
  
  pages={1-1},
  
  doi={10.1109/ACCESS.2021.3071754}}
  
