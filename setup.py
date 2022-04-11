import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lm_eval",
    version="0.2.0",
    author="Leo Gao",
    author_email="lg@eleuther.ai",
    description="A framework for evaluating autoregressive language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EleutherAI/lm-evaluation-harness",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "black",
        "datasets==2.0.0",
        "click>=7.1",
        "scikit-learn>=0.24.1",
        "torch>=1.7",
        "transformers>=4.1",
        "sqlitedict==1.6.0",
        "pytablewriter==0.58.0",
        "sacrebleu==1.5.0",
        "rouge-score==0.0.4",
        "pycountry==20.7.3",
        "numexpr==2.7.2",
        "lm_dataformat==0.0.20",
        "pytest==6.2.3",
        "pybind11==2.6.2",
        "tqdm-multiprocess==0.0.11",
        "zstandard==0.15.2",
        "jsonlines==2.0.0",
        "mock==4.0.3",
        "openai==0.6.4",
        "jieba==0.42.1",
        "nagisa==0.2.7",
        "bleurt@https://github.com/google-research/bleurt/archive/b610120347ef22b494b6d69b4316e303f5932516.zip#egg=bleurt"
    ],
    dependency_links=[
        "https://github.com/google-research/bleurt/archive/b610120347ef22b494b6d69b4316e303f5932516.zip#egg=bleurt",
    ]
)
