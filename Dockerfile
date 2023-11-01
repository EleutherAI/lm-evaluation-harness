FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04


### Install python 3.10 and set it as default python interpreter
RUN  apt update &&  apt install software-properties-common -y && \
add-apt-repository ppa:deadsnakes/ppa -y &&  apt update && \
apt install curl -y && \
apt install python3.10 -y && \
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
apt install python3.10-venv python3.10-dev -y && \
curl -Ss https://bootstrap.pypa.io/get-pip.py | python3.10 && \
apt-get clean && rm -rf /var/lib/apt/lists/


### Copy files
COPY . /lm-evaluation-harness/

### Set working directory

WORKDIR /lm-evaluation-harness


### Install requirements
RUN pip install --no-cache-dir -e .
### Run bash
CMD ["/bin/bash"]
