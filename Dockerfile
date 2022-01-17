ARG TENCENT_ARC_BASE_IMAGE

FROM ${TENCENT_ARC_BASE_IMAGE}

ENV BASICSR_JIT=True
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# weights
ARG TRAINING_FILE1
ENV TRAINING_FILE1 ${TRAINING_FILE1:-v0.2.0/GFPGANCleanv1-NoCE-C2.pth}
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/${TRAINING_FILE1} -P experiments/pretrained_models

ARG TRAINING_FILE2
ENV TRAINING_FILE2 ${TRAINING_FILE2:-v0.1.0/GFPGANv1.pth}
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/${TRAINING_FILE2} -P experiments/pretrained_models

# Copy the entire source following the restrictions of .dockerignore
# Include README if you want to build a library
COPY . .

# https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install/19048754#19048754
RUN python3 setup.py develop
RUN pip3 install realesrgan

ENV CUDA_HOME=/usr/local/cuda

RUN pip3 install basicsr

CMD ["python3", "inference_gfpgan.py", "--model_path", "$${GFPGAN_MODEL_PATH}", "--upscale", "$${GFPGAN_UPSCALE}", "--test_path", "$${GFPGAN_TEST_PATH}", "--save_root", "$${GFPGAN_RESULTS_PATH}"]
