FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get -y install g++ libxrender-dev libxext-dev

COPY requirements.txt .
RUN pip install --upgrade pip

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html

RUN pip install -r requirements.txt
COPY . /GearNet
WORKDIR /GearNet
