FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update && apt-get install -y git
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.9
RUN mkdir /init
COPY ./requirements.txt /init/requirements.txt
RUN python3.9 -m pip -q install pip --upgrade
RUN python3.9 -m pip install -r /init/requirements.txt