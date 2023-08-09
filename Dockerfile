# set the base image
FROM python:3.8-slim-buster 

# set the working directory
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt
CMD ["python3", "application.py"]