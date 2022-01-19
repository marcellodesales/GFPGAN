# GFPGAN (CVPR 2021)

[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/gfpgan)](https://pypi.org/project/gfpgan/)
[![Open issue](https://img.shields.io/github/issues/TencentARC/GFPGAN)](https://github.com/TencentARC/GFPGAN/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/TencentARC/GFPGAN)](https://github.com/TencentARC/GFPGAN/issues)
[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)
[![python lint](https://github.com/TencentARC/GFPGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/TencentARC/GFPGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/publish-pip.yml)

1. [Colab Demo](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo) for GFPGAN <a href="https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>; (Another [Colab Demo](https://colab.research.google.com/drive/1Oa1WwKB4M4l1GmR7CtswDVgOCOeSLChA?usp=sharing) for the original paper model)
2. Online demo: [Huggingface](https://huggingface.co/spaces/akhaliq/GFPGAN) (return only the cropped face)
3. Online demo: [Replicate.ai](https://replicate.com/xinntao/gfpgan) (may need to sign in, return the whole image)
4. We provide a *clean* version of GFPGAN, which can run without CUDA extensions. So that it can run in **Windows** or on **CPU mode**.

> :rocket: **Thanks for your interest in our work. You may also want to check our new updates on the *tiny models* for *anime images and videos* in [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md)** :blush:

GFPGAN aims at developing a **Practical Algorithm for Real-world Face Restoration**.<br>
It leverages rich and diverse priors encapsulated in a pretrained face GAN (*e.g.*, StyleGAN2) for blind face restoration.

:triangular_flag_on_post: **Updates**
- :white_check_mark: Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/GFPGAN).
- :white_check_mark: Support enhancing non-face regions (background) with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
- :white_check_mark: We provide a *clean* version of GFPGAN, which does not require CUDA extensions.
- :white_check_mark: We provide an updated model without colorizing faces.

---

If GFPGAN is helpful in your photos/projects, please help to :star: this repo or recommend it to your friends. Thanks:blush:
Other recommended projects:<br>
:arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration<br>
:arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR): An open-source image and video restoration toolbox<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): A collection that provides useful face-relation functions<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): A PyQt5-based image viewer that is handy for view and comparison<br>

---

### :book: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior

> [[Paper](https://arxiv.org/abs/2101.04061)] &emsp; [[Project Page](https://xinntao.github.io/projects/gfpgan)] &emsp; [Demo] <br>
> [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG

<p align="center">
  <img src="https://xinntao.github.io/projects/GFPGAN_src/gfpgan_teaser.jpg">
</p>

---

# Development

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

## Installating from Github

We now provide a *clean* version of GFPGAN, which does not require customized CUDA extensions. <br>
If you want to use the original model in our paper, please see [PaperModel.md](PaperModel.md) for installation.

1. Clone repo

    ```bash
    git clone https://github.com/TencentARC/GFPGAN.git
    cd GFPGAN
    ```

1. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr

    # Install facexlib - https://github.com/xinntao/facexlib
    # We use face detection and face restoration helper in the facexlib package
    pip install facexlib

    pip install -r requirements.txt
    python setup.py develop

    # If you want to enhance the background (non-face) regions with Real-ESRGAN,
    # you also need to install the realesrgan package
    pip install realesrgan
    ```

## Installing using Docker Containers

This version uses an unnofficial version of Cog Server to implement the interfaces for Machine Learning.

The Cog server from replicated https://github.com/replicate/cog that helps running Machine Learning applications using an API server through a well-defined interface.

> **DOCKER IMAGE**
> * https://hub.docker.com/r/marcellodesales/replicated-cog-server
> * https://github.com/marcellodesales/replicated-cog-server-docker

### What's included

> **DOCS**: More at https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md.

There are a few steps to run you Machine Learning model using cog:

* Created the driver `predict.py`
  * It will define your arguments, their respective types, etc.
  * You will implement the call to your model library
  * You will have an interface to return the types such as images, texts, etc.
* Created the builder `cog.yaml`: It helps describing your dependencies such as system-level, python, and others.
  * System-dependencies: what needs to be in the container to run your model.
  * Model dependencies: pypi dependendencies that is part of your implementation. For instance, the correct versions should be properly described.
  * Pre-install dependencies: Those that are required to be installed after the first ones.
* Docker Artifacts Dockerfile: You can define your dockerfile with the parent image from this repo
* Docker-Compose: It helps keeping all the build parameters for the build

### Building

> NOTE: Make sure to have disk space and memory. (15GB)
> * The first time running it might takes more than 10min depending on your location.
>   * Subsequent Builds take advantage of Docker Caches when specific layers aren't invalidated
> * Problem running: "RGPG invalid signature error while running `apt-get update`": running in MacOS you can have errors like disk space, etc. Just make sure you have enough.
>   * https://stackoverflow.com/questions/64439278/gpg-invalid-signature-error-while-running-apt-update-inside-arm32v7-ubuntu20-04/64553153#64553153

```console
$ docker-compose build
Building GFPGAN
[+] Building 0.2s (18/18) FINISHED
 => [internal] load build definition from Dockerfile                                                                                                          0
 => => transferring dockerfile: 674B                                                                                                                          0
 => [internal] load .dockerignore                                                                                                                             0
 => => transferring context: 35B                                                                                                                              0
 => [internal] load metadata for docker.io/marcellodesales/replicated-cog-server:python3.8_nvidea1.11.1                                                       0
 => [1/3] FROM docker.io/marcellodesales/replicated-cog-server:python3.8_nvidea1.11.1                                                                         0
 => [internal] load build context                                                                                                                             0
 => => transferring context: 4.33kB                                                                                                                           0
 => CACHED [2/3] COPY cog.yaml .                                                                                                                              0
 => CACHED [3/3] RUN cat cog.yaml | yq e . - -o json | jq -r -c '.build.system_packages[]' | sed -r 's/^([^,]*)(,?)$/ \1 \2/' | tr -d '\n' > cog.pkgs &&      0
 => CACHED [4/3] RUN apt-get update -qq && apt-get install -qqy $(cat cog.pkgs) &&     rm -rf /var/lib/apt/lists/* # buildkit 85.8MB buildkit.dockerfile.v0   0
 => CACHED [5/3] RUN cat cog.yaml | yq e . - -o json | jq -r -c '.build.python_packages[]' | sed -r 's/^([^,]*)(,?)$/\1 \2/' | tr -d '\n' > cog.python-pkgs   0
 => CACHED [6/3] RUN pip install -f https://download.pytorch.org/whl/torch_stable.html $(cat cog.python-pkgs)                                                 0
 => CACHED [7/3] RUN cat cog.yaml | yq e . - -o json | jq -r -c '.build.pre_install[]' > cog.pre-inst &&     echo "Installing the pre-install packages: $(ca  0
 => CACHED [8/3] RUN sh cog.pre-inst                                                                                                                          0
 => CACHED [9/3] WORKDIR /src                                                                                                                                 0
 => CACHED [10/3] COPY predict.py .                                                                                                                           0
 => CACHED [11/3] COPY . .                                                                                                                                    0
 => CACHED [12/3] RUN echo "Downloading training file 'v0.2.0/GFPGANCleanv1-NoCE-C2.pth'" &&     wget https://github.com/TencentARC/GFPGAN/releases/download  0
 => CACHED [13/3] RUN echo "Downloading training file 'v0.2.0/GFPGANCleanv1-NoCE-C2.pth'" &&     wget https://github.com/TencentARC/GFPGAN/releases/download  0
 => exporting to image                                                                                                                                        0
 => => exporting layers                                                                                                                                       0
 => => writing image sha256:71684982ed27156781c54ef5e2f7d18a110a7aa0e150bfb49b207e1709102ceb                                                                  0
 => => naming to docker.io/marcellodesales/tencent-arc-gfpgan-runtime                                                                                         0
```

### Running

You can just create a container in the background.

```console
$ docker-compose up -d
Recreating gfpgan_GFPGAN_1 ... done
```

* You can make sure that the container loaded your app and models...

```console
$ docker-compose logs -f
Attaching to gfpgan_GFPGAN_1
GFPGAN_1  | /root/.pyenv/versions/3.8.12/lib/python3.8/site-packages/torch/cuda/__init__.py:52: UserWarning: CUDA initialization: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx (Triggered internally at  /pytorch/c10/cuda/CUDAFunctions.cpp:100.)
GFPGAN_1  |   return torch._C._cuda_getDeviceCount() > 0
GFPGAN_1  | /src/predict.py:41: UserWarning: The unoptimized RealESRGAN is very slow on CPU. We do not use it. If you really want to use it, please modify the corresponding codes.
GFPGAN_1  |   warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
GFPGAN_1  | Downloading: "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" to /root/.pyenv/versions/3.8.12/lib/python3.8/site-packages/facexlib/weights/detection_Resnet50_Final.pth
GFPGAN_1  |
100%|------| 104M/104M [00:04<00:00, 22.7MB/s]
GFPGAN_1  |  * Serving Flask app 'http' (lazy loading)
GFPGAN_1  |  * Environment: production
GFPGAN_1  |    WARNING: This is a development server. Do not use it in a production deployment.
GFPGAN_1  |    Use a production WSGI server instead.
GFPGAN_1  |  * Debug mode: off
GFPGAN_1  |  * Running on all addresses.
GFPGAN_1  |    WARNING: This is a development server. Do not use it in a production deployment.
GFPGAN_1  |  * Running on http://172.19.0.2:5000/ (Press CTRL+C to quit)
```

### Testing Input: HTTP POST image=PATH

* Choose an image as the input to the service.

> Using [viu](https://github.com/atanunq/viu) to open the image on terminal

![Screen Shot 2022-01-18 at 1 56 22 PM](https://user-images.githubusercontent.com/131457/150036554-da9e637b-1b3f-4950-ae18-4b8d236e113e.png)

* Execute the Machine Learning service using the interface built by cog, which exposes the user-defined parameters.
  * In this example, `image` is a parameter

```console
$ curl http://localhost:5000/predict -X POST -F image=@$(pwd)/inputs/whole_imgs/Blake_Lively.jpg -o $(pwd)/super.jpg
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 2087k  100 1996k  100 93345   276k  12943  0:00:07  0:00:07 --:--:--  499k
```

### Testing Output

> Using [viu](https://github.com/atanunq/viu) to open the image on terminal

![Screen Shot 2022-01-18 at 1 56 17 PM](https://user-images.githubusercontent.com/131457/150036575-7f60da84-b89e-4a1a-abcd-084472cebf80.png)

## :zap: Quick Inference

Download pre-trained models: [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth -P experiments/pretrained_models
```

**Inference!**

```bash
python inference_gfpgan.py --upscale 2 --test_path inputs/whole_imgs --save_root results
```

If you want to use the original model in our paper, please see [PaperModel.md](PaperModel.md) for installation and inference.

## :european_castle: Model Zoo

- [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth): No colorization; no CUDA extensions are required. It is still in training. Trained with more data with pre-processing.
- [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth): The paper model, with colorization.

You can find **more models (such as the discriminators)** here: [[Google Drive](https://drive.google.com/drive/folders/17rLiFzcUMoQuhLnptDsKolegHWwJOnHu?usp=sharing)], OR [[Tencent Cloud 腾讯微云](https://share.weiyun.com/ShYoCCoc)]

## :computer: Training

We provide the training codes for GFPGAN (used in our paper). <br>
You could improve it according to your own needs.

**Tips**

1. More high quality faces can improve the restoration quality.
2. You may need to perform some pre-processing, such as beauty makeup.

**Procedures**

(You can try a simple version ( `options/train_gfpgan_v1_simple.yml`) that does not require face component landmarks.)

1. Dataset preparation: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

1. Download pre-trained models and other data. Put them in the `experiments/pretrained_models` folder.
    1. [Pre-trained StyleGAN2 model: StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth)
    1. [Component locations of FFHQ: FFHQ_eye_mouth_landmarks_512.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/FFHQ_eye_mouth_landmarks_512.pth)
    1. [A simple ArcFace model: arcface_resnet18.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth)

1. Modify the configuration file `options/train_gfpgan_v1.yml` accordingly.

1. Training

> python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch

## :whale2: Running GFPGAN in a Docker Container

We provide a docker image of the project for easier installation.

### :wrench: Dependencies and Installation

- [NVIDIA-DOCKER](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU

### Inference

If you want to add new images, follow this convention for directories.
Cropped images in `./inputs/cropped_faces` and
whole images in `./inputs/whole_imgs`

#### v0.1.0

Using <https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth>

```sh
nvidia-docker run \
    --name gfpgan \
    --volume <absolute/path/inputs>:/app/inputs \
    --volume <absolute/path/results>:/app/results \
    {{ DOCKERHUB_REPOSITORY }}/GFPGAN:latest \
    python3 inference_gfpgan.py --model_path experiments/pretrained_models/GFPGANv1.pth --test_path inputs/whole_imgs --save_root results --arch original --channel 1
```

```sh
nvidia-docker run \
    --name gfpgan \
    --volume <absolute/path/inputs>:/app/inputs \
    --volume <absolute/path/results>:/app/results \
    {{ DOCKERHUB_REPOSITORY }}/GFPGAN:latest \
    python3 inference_gfpgan.py --model_path experiments/pretrained_models/GFPGANv1.pth --test_path inputs/cropped_faces --save_root results --arch original --channel 1 --aligned
```

#### v0.2.0

Using <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth>

```sh
nvidia-docker run \
    --name gfpgan \
    --volume <absolute/path/inputs>:/app/inputs \
    --volume <absolute/path/results>:/app/results \
<<<<<<< HEAD
    {{ DOCKERHUB_REPOSITORY }}/GFPGAN:latest \
    python3 inference_gfpgan.py --model_path experiments/pretrained_models/GFPGANv1.pth --test_path inputs/cropped_faces --save_root results --arch original --channel 1 --aligned
```

#### v0.2.0

Using <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth>

```sh
nvidia-docker run \
    --name gfpgan \
    --volume <absolute/path/inputs>:/app/inputs \
    --volume <absolute/path/results>:/app/results \
    {{ DOCKERHUB_REPOSITORY }}/GFPGAN:latest \
    python3 inference_gfpgan.py --upscale 2 --test_path inputs/whole_imgs --save_root results
```

### Training

Follow training steps provided [here](#computer-training) until step 3.

```sh
nvidia-docker run \
    --name gfpgan \
    --volume <absolute/path/experiments/pretrained_models>:/app/experiments/pretrained_models
    --volume <absolute/path/train_gfpgan_v1.yml>:/app/train_gfpgan_v1.yml
    {{ DOCKERHUB_REPOSITORY }}/GFPGAN:latest \
    python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch
```

## :scroll: License and Acknowledgement

GFPGAN is released under Apache License Version 2.0.

## BibTeX

    @InProceedings{wang2021gfpgan,
        author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
        title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }

## :e-mail: Contact

If you have any question, please email `xintao.wang@outlook.com` or `xintaowang@tencent.com`.
