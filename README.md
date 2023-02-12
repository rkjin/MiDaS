## BJ added
1. Test mono depth.
   - run mono_test.py
2. To view realtime monodepth image and to control turtlebot. 
   - upload opencr2.ino  
   - run midasDepthMap.py 
3. Net structure?
   - look midas_dpt_hybrid 
   
고생 끝에...jetson nano 에 

    jetpack 4.6.1 

    $sudo apt update

    $sudo apt install python3-pip

    opencv, tensorrt는 jetpack에 기본으로 포함되어 있음

Pytorch 설치

    $wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl 

    $sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 

    $pip3 install Cython 

    $pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl

torch2trt 설치

    $pip3 install packaging

    $git clone https://github.com/NVIDIA-AI-IOT/torch2trt

    $cd torch2trt

    $sudo python3 setup.py install

torchvision 설치

    $pip3 install pillow

    $sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev 

    $git clone --branch release/0.9 https://github.com/pytorch/vision torchvision  

    $cd torchvision 

    $export BUILD_VERSION=0.9.0 

    $python3 setup.py install --user

scikit-image 설치

    $sudo apt-get install liblapack-dev gfortran

    scipy 설치

       $wget https://github.com/scipy/scipy/releases/download/v1.3.3/scipy-1.3.3.tar.gz

       $tar -xzvf scipy-1.3.3.tar.gz scipy-1.3.3

       $cd scipy-1.3.3/

       $python3 setup.py install --user

    Tiff 설치

       $wget https://download.osgeo.org/libtiff/tiff-4.1.0.tar.gz

       $tar -xzvf tiff-4.1.0.tar.gz

       $cd tiff-4.1.0/

       $./configure
 
       $make

       $sudo make install

    scikit-image 설치

       $sudo apt-get install python3-sklearn

       $sudo apt-get install libaec-dev libblosc-dev libffi-dev libbrotli-dev libboost-all-dev libbz2-dev

       $sudo apt-get install libgif-dev libopenjp2-7-dev liblcms2-dev libjpeg-dev libjxr-dev liblz4-dev liblzma-dev libpng-dev libsnappy-dev libwebp-dev libzopfli-dev libzstd-dev

       $sudo pip3 install imagecodecsp-dev libzopfli-dev libzstd-dev

       $sudo pip3 install scikit-image

테스트

   $python3  
     import torch  
     import tensorrt  
     import torch2trt  
   
   
<br>
<a href="default.asp"><img src=https://user-images.githubusercontent.com/87571989/198910800-8fbc3b43-959a-4435-85ce-93292ff50c37.png></img> </a>

## Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer

This repository contains code to compute depth from a single image. It accompanies our [paper](https://arxiv.org/abs/1907.01341v3):

>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer  
René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun


and our [preprint](https://arxiv.org/abs/2103.13413):

> Vision Transformers for Dense Prediction  
> René Ranftl, Alexey Bochkovskiy, Vladlen Koltun


MiDaS was trained on 10 datasets (ReDWeb, DIML, Movies, MegaDepth, WSVD, TartanAir, HRWSI, ApolloScape, BlendedMVS, IRS) with
multi-objective optimization. 
The original model that was trained on 5 datasets  (`MIX 5` in the paper) can be found [here](https://github.com/intel-isl/MiDaS/releases/tag/v2).


### Changelog
* [Sep 2021] Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/DPT-Large).
* [Apr 2021] Released MiDaS v3.0:
    - New models based on [Dense Prediction Transformers](https://arxiv.org/abs/2103.13413) are on average [21% more accurate](#Accuracy) than MiDaS v2.1
    - Additional models can be found [here](https://github.com/intel-isl/DPT)
* [Nov 2020] Released MiDaS v2.1:
	- New model that was trained on 10 datasets and is on average about [10% more accurate](#Accuracy) than [MiDaS v2.0](https://github.com/intel-isl/MiDaS/releases/tag/v2)
	- New light-weight model that achieves [real-time performance](https://github.com/intel-isl/MiDaS/tree/master/mobile) on mobile platforms.
	- Sample applications for [iOS](https://github.com/intel-isl/MiDaS/tree/master/mobile/ios) and [Android](https://github.com/intel-isl/MiDaS/tree/master/mobile/android)
	- [ROS package](https://github.com/intel-isl/MiDaS/tree/master/ros) for easy deployment on robots
* [Jul 2020] Added TensorFlow and ONNX code. Added [online demo](http://35.202.76.57/).
* [Dec 2019] Released new version of MiDaS - the new model is significantly more accurate and robust
* [Jul 2019] Initial release of MiDaS ([Link](https://github.com/intel-isl/MiDaS/releases/tag/v1))

### Setup 

1) Pick one or more models and download corresponding weights to the `weights` folder:

- For highest quality: [dpt_large](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt)
- For moderately less quality, but better speed on CPU and slower GPUs: [dpt_hybrid](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt)
- For real-time applications on resource-constrained devices: [midas_v21_small](https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt)
- Legacy convolutional model: [midas_v21](https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt)

2) Set up dependencies: 

    ```shell
    conda install pytorch torchvision opencv
    pip install timm
    ```

   The code was tested with Python 3.7, PyTorch 1.8.0, OpenCV 4.5.1, and timm 0.4.5.

    
### Usage

1) Place one or more input images in the folder `input`.

2) Run the model:

    ```shell
    python run.py --model_type dpt_large
    python run.py --model_type dpt_hybrid 
    python run.py --model_type midas_v21_small
    python run.py --model_type midas_v21
    ```

3) The resulting inverse depth maps are written to the `output` folder.


#### via Docker

1) Make sure you have installed Docker and the
   [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-\(Native-GPU-Support\)).

2) Build the Docker image:

    ```shell
    docker build -t midas .
    ```

3) Run inference:

    ```shell
    docker run --rm --gpus all -v $PWD/input:/opt/MiDaS/input -v $PWD/output:/opt/MiDaS/output midas
    ```

   This command passes through all of your NVIDIA GPUs to the container, mounts the
   `input` and `output` directories and then runs the inference.

#### via PyTorch Hub

The pretrained model is also available on [PyTorch Hub](https://pytorch.org/hub/intelisl_midas_v2/)

#### via TensorFlow or ONNX

See [README](https://github.com/intel-isl/MiDaS/tree/master/tf) in the `tf` subdirectory.

Currently only supports MiDaS v2.1. DPT-based models to be added. 


#### via Mobile (iOS / Android)

See [README](https://github.com/intel-isl/MiDaS/tree/master/mobile) in the `mobile` subdirectory.

#### via ROS1 (Robot Operating System)

See [README](https://github.com/intel-isl/MiDaS/tree/master/ros) in the `ros` subdirectory.

Currently only supports MiDaS v2.1. DPT-based models to be added. 


### Accuracy

Zero-shot error (the lower - the better) and speed (FPS):

| Model |  DIW, WHDR | Eth3d, AbsRel | Sintel, AbsRel | Kitti, δ>1.25 | NyuDepthV2, δ>1.25 | TUM, δ>1.25 | Speed, FPS |
|---|---|---|---|---|---|---|---|
| **Small models:** | | | | | | | iPhone 11 |
| MiDaS v2 small | **0.1248** | 0.1550 | **0.3300** | **21.81** | 15.73 | 17.00 | 0.6 |
| MiDaS v2.1 small [URL]() | 0.1344 | **0.1344** | 0.3370 | 29.27 | **13.43** | **14.53** | 30 |
| | | | | | | |
| **Big models:** | | | | | | | GPU RTX 3090 |
| MiDaS v2 large [URL](https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt) | 0.1246 | 0.1290 | 0.3270 | 23.90 | 9.55 | 14.29 | 51 |
| MiDaS v2.1 large [URL](https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt) | 0.1295 | 0.1155 | 0.3285 | 16.08 | 8.71 | 12.51 | 51 |
| MiDaS v3.0 DPT-Hybrid [URL](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt) | 0.1106 | 0.0934 | 0.2741 | 11.56 | 8.69 | 10.89 | 46 |
| MiDaS v3.0 DPT-Large [URL](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt) | **0.1082** | **0.0888** | **0.2697** | **8.46** | **8.32** | **9.97** | 47 |



### Citation

Please cite our paper if you use this code or any of the models:
```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```

If you use a DPT-based model, please also cite:

```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
```

### Acknowledgements

Our work builds on and uses code from [timm](https://github.com/rwightman/pytorch-image-models). 
We'd like to thank the author for making these libraries available.

### License 

MIT License 
