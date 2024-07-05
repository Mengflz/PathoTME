from setuptools import setup, find_packages

setup(
    name='PathoTME',
    version='0.1.0',
    description='PathoTME: A deep learning framework for predicting the tumor microenvironment in histopathology images.',
    url='https://github.com/mengflz/PathoTME',
    author='FLZM,HRZ,RDY',
    author_email='mengflz@outlook.com',
    packages=find_packages(),
    install_requires=["torch>=1.12.1", "gseapy",
                      "numpy", "pandas", "scikit-learn", "tqdm", "pickle", "yaml"
                      "argparse", "glob", "statistics"],
    python_requires='>=3.7',
    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: CC BY-NC 4.0",
]
)