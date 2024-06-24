FROM pytorch/pytorch:latest
WORKDIR /usr/src/app

# Copy the project files into the container
COPY . .

# Install dependencies
RUN pip install -e ".[dev]"
# Define the default command
RUN apt update
RUN apt install git -y
RUN pip install -U torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip uninstall numpy -y
RUN pip install numpy==1.26.4
RUN pip install sentencepiece protobuf
CMD /bin/bash
