import os

from setuptools import find_packages, setup

# read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


requirements = [
    "gymnasium",
    "matplotlib",
    "numpy",
    "pandas",
    "pygame",
    "shapely",
    "svgpath2mpl",
]

setup(
    name="mobile-env",
    version="2.0.1",
    author="Stefan Schneider, Stefan Werner",
    description="mobile-env: An Open Environment for Autonomous Coordination in "
    "Wireless Mobile Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanbschneider/mobile-env",
    packages=find_packages(),
    python_requires=">=3.7.0",
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
