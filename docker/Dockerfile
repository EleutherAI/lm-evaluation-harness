FROM ubuntu:24.04

RUN apt-get update \
  && apt-get install -y python3 python3-pip python3-venv \
  && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

COPY . /lm-evaluation-harness
WORKDIR /lm-evaluation-harness
RUN mkdir output \
  && chmod u+x /lm-evaluation-harness/docker/install_submodules.sh \
  && chmod u+x /lm-evaluation-harness/docker/install_packages.sh

# enable venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# prepare lm_eval and extra packages as defined by user
ARG submodules
ARG extra_packages
RUN python3 -m pip install --upgrade --no-cache-dir pip \
  && /lm-evaluation-harness/docker/install_submodules.sh ${submodules} \
  && /lm-evaluation-harness/docker/install_packages.sh ${extra_packages}

CMD ["lm_eval",  \
      "--model", "hf", \
      "--model_args", "pretrained=EleutherAI/pythia-14m", \
      "--tasks", "ai2_arc", \
      "--device", "cpu", \
      "--output_path", "./output", \
      "--log_samples"]
