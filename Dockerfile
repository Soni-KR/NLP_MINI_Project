FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY finrag.py evaluate.py test_cases.json rapport_biat_2021.pdf rapport_biat_2023.pdf rapport_biat_2025.pdf ./

ENTRYPOINT ["python", "/app/finrag.py"]
