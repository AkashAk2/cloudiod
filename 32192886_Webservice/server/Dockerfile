FROM python:3.9-slim-buster
WORKDIR /server
COPY . /server
RUN apt-get update -y && apt-get install -y\
 python3-pip python3-dev libsm6 libxext6 \
 libgl1 libxrender-dev libglib2.0-0 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python3", "/server/object_detection.py"]