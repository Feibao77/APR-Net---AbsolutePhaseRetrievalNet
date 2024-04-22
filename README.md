# APR-Net: Deep learning-based frequency-multiplexing composite-fringe projection profilometry technique for one-shot 3D shape measurement
<div align="center">
 
**Yifei Chen, Jiehu Kang, Luyuan Feng, Leiwen Yuan, Jian Liang, Zongyang Zhao, Bin Wu**
**the State Key Laboratory of Precision Measurement Technology and Instruments, Tianjin University, China**
This is our [official published article]().
 
</div>

## Abstract
The pursuit of precise and efficient 3D shape measurement has long been a focal point within the fringe projection profilometry (FPP) domain. However, achieving precise 3D reconstruction for isolated objects from a single fringe image remains a challenging task in the field. In this paper, a deep learning-based frequency-multiplexing (FM) composite-fringe projection profilometry (DFCP) is proposed, where an end-to-end absolute phase retrieval network (APR-Net) is trained to directly recover the absolute phase map from a FM composite fringe pattern. The obtained absolute phase map exhibits exceptional precision and is devoid of spectrum crosstalk or leakage disturbance typically encountered in traditional FM techniques. APR-Net is intricately crafted, incorporating a nested strategy along with the concept of centralized information interaction. A diverse dataset encompassing various scenarios and free from spectrum aliasing is assembled to guide the training process. In the first qualitative experiment, DFCP demonstrates comparable phase accuracy to ground truths with 47 fewer projected images. The second qualitative experiment and the quantitative evaluation respectively prove DFCP’s capability in high dynamic range 3D measurement and in precise 3D measurement.

## Architecture of APR-Net

![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/59141cb3-2b73-4161-a73c-8692bc1e3d5c)

**Fig.1.** Schematic of APR-Net.


## Dataset
We have built a [dataset](https://drive.google.com/file/d/1G_cBiRRJErjvl2iE-Ga8_T3IouuziJPH/view?usp=drive_link) from scratch.

## Experiments and results

![image](https://github.com/Feibao77/APR-Net---AbsolutePhaseRetrievalNet/assets/117697608/d83ead7b-6160-49fc-9b20-c6525d8821cd)

**Fig. 2.** Comparison between the 3D reconstruction results measured by five methods. (a, b, c) The 3D results by conventional tri-frequency Fourier Transform method. (d, e, f) The 3D results by [Nguyen's](https://www.sciencedirect.com/science/article/pii/S0263224121015281) method. (g, h, i) The 3D results by U-Net-based method. (j, k, l) The 3D results by our method. (m, n, o) The 3D results by ground-truth generation method (16-step PSP with tri-frequency TPU). 

## Citation
