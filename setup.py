import os

from setuptools import find_packages, setup

# read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


requirements = [
    # gymnasium 1.0 introduced breaking changes (see docs/components.md); <2.0 to match the
    # range that stable-baselines3 and Ray RLlib (see tests/requirements.txt) support
    "gymnasium>=0.29.1,<2.0",
    "matplotlib>=3.5",
    "numpy>=1.22",
    "pandas>=1.5",
    "pygame>=2.1",
    "shapely>=2.0",
    "svgpath2mpl>=1.0",
]

setup(
    name="mobile-env",
    version="2.1.0",
    author="Stefan Schneider, Stefan Werner",
    description="mobile-env: An Open Environment for Autonomous Coordination in "
    "Wireless Mobile Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanbschneider/mobile-env",
    packages=find_packages(),
    python_requires=">=3.9.0",
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
