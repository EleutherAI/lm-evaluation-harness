from setuptools import setup, find_packages
from setuptools.command.install import install


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


dev_requires = (["black<=21.12b0", "coverage<=6.2", "mock>=4.0.3", "pytest"],)
install_requires = [
    "datasets>=2.0.0",
    "codecarbon",
    "nltk==3.6",
    "openai==0.13.0",
    "pycountry==20.7.3",
    "pytablewriter==0.58.0",
    "rouge-score==0.0.4",
    "sacrebleu==1.5.0",
    "scikit-learn>=0.24.1",
    "sqlitedict==1.6.0",
    "torch>=1.9",
    "tqdm-multiprocess==0.0.11",
    "accelerate@git+https://github.com/huggingface/accelerate@main",
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
    python_requires=">=3.8",
    install_requires=install_requires,
    dependency_links=dependency_links,
    extras_require={"dev": dev_requires},
    cmdclass={"install": PostInstall},
)
