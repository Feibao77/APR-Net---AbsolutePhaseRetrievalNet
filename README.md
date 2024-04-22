# APR-Net---AbsolutePhaseRetrievalNet
If you need more information, see our [paper]().

## Brief Introduction
APR-Net enables pixel-wise prediction of the absolute phase map from a single deformed frequency-multiplexing composite image. A nested strategy leveraging Residual U-block (RSU) and the concept of centralized information interaction (CII) are employed.

## Architecture of APR-Net
The backbone of APR-Net adopts two levels of U-shape structure (see Fig. 1). On the exterior level, the U-shape structure consists of several encoder modules and decoder modules. On the interior level, each exterior encoder or decoder is a RSU (see Fig. 2) which is also a U-shape structure. The nested U-shape structure facilitates hierarchical feature extraction, allowing the network to capture multi-scale information inherent in the input. 

![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/59141cb3-2b73-4161-a73c-8692bc1e3d5c)
Fig.1. Schematic of APR-Net.

![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/1ea2698c-251b-4d06-b378-f9c250ab46f2)
Fig. 2. Schematic of RSU. (a) RSU-L. (b) RSU-4F.

Additionally, rather than using a direct identity mapping from an exterior encoder output to the corresponding decoder input, we incorporate a relative global calibration (RGC) module (see Fig. 3) into the network to strengthen the information interaction among feature maps in diverse resolutions. 
It should be noted that there is only one learnable RGC module in APR-Net, meaning the five RGC modules displayed in Fig. 1 share the same set of filter parameters. Thus, not only En_1-En_5 achieve information interaction with En_6 via RGC, but also En_1-En_5 interact with each other due to the parameters-sharing mechanism. The detailed configurations of each encoder and decoder can be seen in our paper.

![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/78f780e3-3ee5-4c9d-8e70-0cad55f14ea7)
Fig. 3. Architecture of RGC.

## Experiments and results




