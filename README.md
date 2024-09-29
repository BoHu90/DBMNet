# DBMNet

[1] Bo Hu, Shuaijian Wang, Xinbo Gao, Leida Li, Ji Gan, and Xixi Nie, "Reduced-Reference Image Deblurring Quality Assessment Based on Multiscale Feature Enhancement and Aggregation," Neurocomputing, vol. 549, 2023, p. 126501.

The paper is available on [website](https://www.sciencedirect.com/science/article/pii/S0925231223005015?via%3Dihub).

# Abstract

Image deblurring is a basic task in the field of computer vision, and has attracted much attention because of its application prospects in traffic monitoring and medical imaging, etc. Due to the inherent weakness of the model, it is difficult to obtain well-pleasing deblurred images for all the visual contents so far. Therefore, how to objectively evaluate the quality of these deblurred results is very important for the rapid development of image deblurring. In recent years, numerous convolutional neural networks based quality assessment methods have been proposed to automatically predict the quality of synthetic and authentic distorted images, producing results that are mildly consistent with subjective perception. However, they are limited in Image Deblurring Quality Assessment (IDQA). For IDQA, it is more meaningful to predict the quality difference of blurry-deblurred image (BDI) pair than to make prediction on single deblurred image. Inspired by this, we propose a novel reduced-reference image deblurring quality assessment method based on multi-scale feature enhancement and aggregation. Firstly, the multi-scale features of BDI pair are generated from a versatile vision Transformer. Secondly, the discrepancy information is exploited to implicitly enhance the initial deep features. Finally, the enhanced features of different scales are aggregated and then mapped to the quality difference of BDI pair. Experimental results on four challenging datasets demonstrate that the proposed method is superior to the state-of-the-art quality assessment methods.

# Dependencies

- numpy
- pandas
- Pillow
- scipy
- timm
- torch
- torchvision
- tqdm

More information please check the `requirements.txt`.