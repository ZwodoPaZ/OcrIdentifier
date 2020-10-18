FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y tesseract-ocr-swe python3-pip
WORKDIR ocr_identifier
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
USER 1001
