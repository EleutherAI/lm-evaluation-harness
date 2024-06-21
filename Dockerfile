FROM pytorch/pytorch:latest
WORKDIR /usr/src/app

# Copy the project files into the container
COPY . .

# Install dependencies
RUN pip install -e ".[dev]"
# Define the default command
RUN apt update
RUN apt install git -y
CMD /bin/bash

lm_eval --model hf \
    --model_args pretrained=mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tasks halftruthdetection \
    --device cuda:0 \
    --batch_size 8