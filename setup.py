import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lm_eval",
    version="1.0.0",
    author="EleutherAI",
    author_email="contact@eleuther.ai",
    description="A framework for evaluating language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EleutherAI/lm-evaluation-harness",
    packages=setuptools.find_packages(),
    # required to include yaml files in pip installation
    package_data={
        "lm_eval": ["**/*.yaml", "tasks/**/*"],
        "examples": ["**/*.yaml"],
    },
    entry_points={
        "console_scripts": ["lm-eval = main:main", "lm_eval = main:main"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "accelerate>=0.18.0",
        "evaluate",
        "datasets>=2.0.0",
        "evaluate>=0.4.0",
        "jsonlines",
        "numexpr",
        "openai>=0.6.4",
        "omegaconf>=2.2",
        "peft>=0.2.0",
        "pybind11>=2.6.2",
        "pycountry",
        "pytablewriter",
        "rouge-score>=0.0.4",
        "sacrebleu==1.5.0",
        "scikit-learn>=0.24.1",
        "sqlitedict",
        "torch>=1.8",
        "tqdm-multiprocess",
        "transformers>=4.1",
        "zstandard",
    ],
    extras_require={
        "dev": ["black", "flake8", "pre-commit", "pytest", "pytest-cov"],
        "linting": [
            "flake8",
            "pylint",
            "mypy",
            "pre-commit",
        ],
        "testing": ["pytest", "pytest-cov", "pytest-xdist"],
        "multilingual": ["nagisa>=0.2.7", "jieba>=0.42.1"],
        "sentencepiece": ["sentencepiece>=0.1.98", "protobuf>=4.22.1"],
        "promptsource": [
            "promptsource @ git+https://github.com/bigscience-workshop/promptsource.git#egg=promptsource"
        ],
        "gptq": ["auto-gptq[triton] @ git+https://github.com/PanQiWei/AutoGPTQ"],
        "anthropic": ["anthropic"],
    },
)
