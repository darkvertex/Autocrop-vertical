# Runpod Serverless image for AutoCrop-Vertical.
# Replicate builds from cog.yaml (`cog build`). Cog does not use this file
# unless `cog build --dockerfile` is passed, and CI does not pass that flag.
#
# GPU is for NVENC. PyTorch stays on the CPU wheel, matching the Cog image.
# NVIDIA's entrypoint is kept so the host driver and libnvidia-encode are wired
# up. NVIDIA_DRIVER_CAPABILITIES=all is what makes h264_nvenc / hevc_nvenc open.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/src \
    YOLO_WEIGHTS=/src/models/yolov8n.pt \
    YOLO_OFFLINE=1 \
    NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        ffmpeg \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /src

COPY replicate-requirements.txt /src/replicate-requirements.txt
RUN pip install --no-cache-dir -r /src/replicate-requirements.txt \
    && pip install --no-cache-dir "runpod~=1.7"

RUN mkdir -p /src/models \
    && curl -fsSL -o /src/models/yolov8n.pt \
        https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt \
    && test "$(stat -c%s /src/models/yolov8n.pt)" -gt 6000000

COPY main.py tracking.py predict.py handler.py /src/

CMD ["python", "-u", "/src/handler.py"]
