# RunPod-friendly CUDA image (PyTorch pre-installed)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /workspace

COPY service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY service/ /workspace/service/

# Mount artifacts at /workspace/artifacts (probe_config.json lives there).
# The probe config references the HF model id; weights download on first boot.
ENV PROBE_CONFIG=/workspace/artifacts/probe_config.json

EXPOSE 8000

CMD ["python", "service/server.py"]
