FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY app.py ./

RUN pip install --no-cache-dir -e .

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
CMD ["python", "app.py"]
