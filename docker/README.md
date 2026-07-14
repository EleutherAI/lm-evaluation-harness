# Build and Run an `lm-evaluation-harness` Docker Image

This subfolder contains the Dockerfile and instructions for building and running a Docker image to quickly get started with `lm-evaluation-harness`.

## Prerequisites

- Make sure you have Docker (or another container tool platform that aliases to `docker`) CLI tools installed
- Clone this repository to your local machine:

    ```bash
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    ```

## Running a Simple Example

### Building the image

1. From the repository root, build the Docker image using the following command:

    ```bash
    docker build --tag [image_name]:[tag] --file ./docker/Dockerfile .
    ```

    Replace `[image_name]` with a name of your choice for the image, and `[tag]` with a version or description (e.g., `latest`, `v1.0.0`, etc.).

    The `--file` option tells docker where to look for the Dockerfile relative to our current directory, which is the repository root (signified with the `.` at the very end of the command).

2. Verify that the image was built successfully by listing all local images:

    ```bash
    docker images
    ```

    You should see your new image listed in the output.

### Running the container

1. Run a container using the built image with the following command:

    ```bash
    docker run -it --name [container_name] [image_name]:[tag]
    ```

    Replace `[container_name]` with a name of your choice for the running container. Replace `[image_name]:[tag]` with the value you supplied during the build.

    A simple example evaluation will kick off that runs the following command:

    ```bash
    lm_eval
        --model hf
        --model_args "pretrained=EleutherAI/pythia-14m"
        --tasks ai2_arc
        --device cpu
        --output_path ./output
        --log_samples
    ```

    You can track the progress via the console logs. If you would like the output of the evaluation job to be available on your local machine, mount a volume to the container, as described in the [advanced usage](#advanced-usage) section.

2. Once the job finishes, you can remove the container with the following commands:

    ```bash
    docker rm [container_name]
    ```

## Customizing the Image Environment and/or the Evaluation Job

The Dockerfile is set up to allow you to customize the image and container run parameters, each with a single command.

### Building the image with additional packages

1. The Dockerfile includes the ability to specify two arguments at build time that allow you to customize the environment for an evaluation run:

   - `submodules`: a comma-separated list of `lm_eval` package extras that you want to be installed in the build environment (e.g. `--build-arg submodules=ibm_watsonx_ai,longbench`)
   - `extra_packages`: a comma-separated list of additional PyPi packages that you want to be installed in the build environment (e.g. `--build-arg extra_packages=unitxt,requests`). You can also specify required versions as necessary (e.g. `--build-arg extra_packages=unitxt>=1.19.0`)

    From the repository root, build the Docker image using the following command:

    ```bash
    docker build --tag [image_name]:[tag] --file ./docker/Dockerfile --build-arg submodules=extra1,extra2 --build-arg extra_packages=package1,package2 .
    ```

    Replace `[image_name]` with a name of your choice for the image, and `[tag]` with a version or description (e.g., `latest`, `v1.0.0`, etc.).

    The `--file` option tells docker where to look for the Dockerfile relative to our current directory, which is the repository root.

2. Verify that the image was built successfully by listing all local images:

    ```bash
    docker images
    ```

    You should see your new image listed in the output.

### Running the container with a different lm_eval command

1. The image Dockerfile includes the ability to specify custom arguments to the `lm_eval` command.

    You can run a container using custom arguments using the following command:

    ```bash
    docker run -it --name [container_name] [image_name]:[tag] lm_eval --[arg1] [value] --[arg2] [value]
    ```

    Replace `[container_name]` with a name of your choice for the running container. Replace `[image_name]:[tag]` with the value you supplied during the build. Everything after the image name and tag is the new command that we want to run when the container starts. Custom evaluation parameters should appear after the `lm_eval` command.

    Example: `docker run -it --name flan-t5-eval my_lm_eval_image:latest lm_eval --model hf --model_args "pretrained=google/flan-t5-base" --tasks "arc_easy" --device cpu --limit 10`

2. Once the job finishes, you can remove the container with the following commands:

    ```bash
    docker rm [container_name]
    ```

## Advanced Usage

### Manually run `lm_eval` CLI commands

There are several situations where it may be more useful to manually run `lm_eval` commands, set environment variables, and download packages within the container.

1. To do so, start a container with the `bash` shell.

    ```bash
    docker run -it --name [container_name] [image_name]:[tag] /bin/bash
    ```

    You will see a prompt in the terminal signaling that you are now in a shell in the container:

    ```bash
    root@765037aee64f:/lm-evaluation-harness#
    ```

2. Once inside the container, you can perform any necessary operations. Kick off an evaluation job with an `lm_eval` command:

    ```bash
    lm_eval --model hf --model_args "pretrained=google/flan-t5-base" --tasks "arc_easy" --limit 10 --device cpu --output ./output
    ```

    By running the commands with this message, you can now inspect output results without the need to mount a volume (below).

### Use a volume to persist the output of an evaluation run on your local machine

To ensure that the output of an evaluation is available on your local machine, you can mount a volume into the container:

```bash
docker run -it -v [path/to/local/directory]:/lm-evaluation-harness/output --name [container_name] [image_name]:[tag]
```

If you have modified the path to the `--output` parameter of the `lm_eval` command, make sure you provide the correct absolute path on the right side of the colon `:` instead of the default `/lm-evaluation-harness/output`.

### Supply environment variables to the container

If your container needs access to certain evnironment variables, pass them with the `-e` option to the `docker run` command:

```bash
docker run -it -e ENV_VAR1=value1 -e ENV_VAR2=value2 --name [container_name] [image_name]:[tag]
```

### Run the container in the background

Use the `-d` option in the `docker run` command to run the container in the background, leaving the terminal available to you for other commands.

```bash
docker run -it -d --name [container_name] [image_name]:[tag]
```

### Modify the `.dockerignore` file

This Dockerfile copies everything from the root repository directory into the image, except for any paths specified in the `.dockerignore` file. Excluding these files is necessary to keep the size of the image/container as small as possible and to avoid any out of memory errors. If you have created any large directories or files anywhere in the file structure starting at the repository root that you don't want copied into the image, edit the `.dockerignore` file to direct it to avoid copying these files into the container.

### Additional Resources

- Docker documentation: https://docs.docker.com/
- Dockerfile reference: https://docs.docker.com/engine/reference/builder/
