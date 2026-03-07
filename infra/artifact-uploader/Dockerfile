FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir boto3==1.35.99

COPY uploader.py /app/uploader.py

CMD ["python", "/app/uploader.py"]
