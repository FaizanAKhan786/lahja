FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -e .

EXPOSE 8000
ENV LAHJA_MODEL_DIR=/models
# For GPU inference, build with the train extra: pip install -e ".[train]"
CMD ["uvicorn", "lahja.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
