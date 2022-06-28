from setuptools import setup, find_packages
from setuptools.command.install import install


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


dev_requires = (["black", "coverage", "mock>=4.0.3", "pytest"],)
install_requires = [
    "datasets",
    "codecarbon",
    "jsonlines==2.0.0",
    "lm_dataformat==0.0.20",
    "nltk",
    "openai",
    "pycountry==20.7.3",
    "pybind11==2.6.2",
    "pytablewriter==0.58.0",
    "rouge-score==0.0.4",
    "sacrebleu==1.5.0",
    "scikit-learn>=0.24.1",
    "sqlitedict==1.6.0",
    "torch>=1.7",
    "tqdm-multiprocess==0.0.11",
    "transformers@git+https://github.com/huggingface/transformers@main",
    "promptsource@git+https://github.com/bigscience-workshop/promptsource@eval-hackathon",
]
dependency_links = []


class PostInstall(install):
    @staticmethod
    def post_install():
        """Post installation `nltk` downloads."""
        import nltk

        nltk.download("popular")

    def run(self):
        install.run(self)
        self.execute(
            PostInstall.post_install, [], msg="Running post installation tasks"
        )


setup(
    name="lm_eval",
    version="0.2.0",
    author="Leo Gao & EleutherAI",
    description="A framework for evaluating autoregressive language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EleutherAI/lm-evaluation-harness",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
    dependency_links=dependency_links,
    extras_require={"dev": dev_requires},
    cmdclass={"install": PostInstall},
)
