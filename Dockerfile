ARG TENCENT_ARC_BASE_IMAGE
ARG TRAINING_FILE1
ARG TRAINING_FILE2

FROM ${TENCENT_ARC_BASE_IMAGE}

# weights
ARG TRAINING_FILE1
ENV TRAINING_FILE1 ${TRAINING_FILE1:-v0.2.0/GFPGANCleanv1-NoCE-C2.pth}
RUN echo "Downloading training file '${TRAINING_FILE1}'" && \
    wget https://github.com/TencentARC/GFPGAN/releases/download/${TRAINING_FILE1} -P experiments/pretrained_models

ARG TRAINING_FILE2
ENV TRAINING_FILE2 ${TRAINING_FILE2:-v0.1.0/GFPGANv1.pth}
RUN echo "Downloading training file '${TRAINING_FILE1}'" && \
    wget https://github.com/TencentARC/GFPGAN/releases/download/${TRAINING_FILE2} -P experiments/pretrained_models
