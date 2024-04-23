# APR-Net: Deep learning-based frequency-multiplexing composite-fringe projection profilometry technique for one-shot 3D shape measurement
<div align="center">
 
**Yifei Chen, Jiehu Kang, Luyuan Feng, Leiwen Yuan, Jian Liang, Zongyang Zhao, Bin Wu**

**State Key Laboratory of Precision Measurement Technology and Instruments, Tianjin University, China**
</div>

This is our [official published article]().

## Abstract
The pursuit of precise and efficient 3D shape measurement has long been a focal point within the fringe projection profilometry (FPP) domain. However, achieving precise 3D reconstruction for isolated objects from a single fringe image remains a challenging task in the field. In this paper, a deep learning-based frequency-multiplexing (FM) composite-fringe projection profilometry (DFCP) is proposed, where an end-to-end absolute phase retrieval network (APR-Net) is trained to directly recover the absolute phase map from a FM composite fringe pattern. The obtained absolute phase map exhibits exceptional precision and is devoid of spectrum crosstalk or leakage disturbance typically encountered in traditional FM techniques. APR-Net is intricately crafted, incorporating a nested strategy along with the concept of centralized information interaction. A diverse dataset encompassing various scenarios and free from spectrum aliasing is assembled to guide the training process. In the first qualitative experiment, DFCP demonstrates comparable phase accuracy to ground truths with 47 fewer projected images. The second qualitative experiment and the quantitative evaluation respectively prove DFCP’s capability in high dynamic range 3D measurement and in precise 3D measurement.

## Architecture of APR-Net
<div align="center">
 
![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/59141cb3-2b73-4161-a73c-8692bc1e3d5c)

</div>

**Fig.1.** Schematic of APR-Net.



## Dataset
We have built a [dataset](https://drive.google.com/file/d/1FPXVvhIQrH3uUldxcCWBrBJbryDF0NC1/view?usp=sharing) from scratch. We have uploaded only 500 training set samples and 53 validation set samples. If you need the complete dataset, please contact us.

<div align="center">
 
![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/602e0f37-8091-4768-bf02-9599853a488a)

![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/c85f6efa-b4cf-4818-83ca-a0d049d11bfd)

</div>

**Fig. 2.** Dataset. (a) Flowchart of ground-truth generation (16-step PSP with tri-frequency TPU). (b) Several sets of samples.

## Experiments and results
### Qualitative evaluation

<div align="center">
 
![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/d83ead7b-6160-49fc-9b20-c6525d8821cd)

</div>

**Fig. 3.** Comparison between the 3D reconstruction results measured by five methods. (a, b, c) The 3D results by conventional tri-frequency Fourier Transform method. (d, e, f) The 3D results by [Nguyen's](https://www.sciencedirect.com/science/article/pii/S0263224121015281) method. (g, h, i) The 3D results by U-Net-based method. (j, k, l) The 3D results by our method. (m, n, o) The 3D results by ground-truth generation method (16-step PSP with tri-frequency TPU). 


 
**Table 1. Comparison of five methods**

<img width="585" alt="e6e371d6e07fc58b0290024f5b4cc48" src="https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/a457caa0-f7e7-4caf-a46a-3b4798658148">


### Quantitative evaluation

<div align="center">
 
![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/95018939-df1c-4815-85c3-36dd41ccff31)

</div>

**Fig. 4** Measurement results of a pair of standard spheres by DFCP. (a) 3D results. (b) 3D error distribution.








**Table 2.** Measurement values of a pair of standard spheres by DFCP
<img width="574" alt="image" src="https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/e52cc922-fb16-467f-a6ba-524fed5bbe89">


