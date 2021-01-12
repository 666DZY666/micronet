import os

from setuptools import setup, find_packages

from micronet import __version__

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    install_requires=install_requires,
    name="micronet",
    version=__version__,
    author="666DZY666",
    author_email="dzy_pku@pku.edu.cn",
    description="A model compression and deploy lib.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/666DZY666/micronet",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
