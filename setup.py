import os
from setuptools import setup, find_packages


# read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


requirements = [
    "gym>=0.19.0",
    "matplotlib==3.5.0",
    "numpy==1.21.4",
    "pygame==2.1.0",
    "shapely==1.8.0",
    "svgpath2mpl==1.0.0",
]

setup(
    name="mobile-env",
    version="1.0.0",
    author="Stefan Schneider, Stefan Werner",
    description="mobile-env: An Open Environment for Autonomous Coordination in Wireless Mobile Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanbschneider/mobile-env",
    packages=find_packages(),
    python_requires=">=3.7.*",
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)