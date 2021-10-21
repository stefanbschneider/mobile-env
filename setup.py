import os
from setuptools import setup, find_packages


# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


requirements = [
    'shapely>=1.7.0',
    'matplotlib>=3.4.3',
    'numpy>=1.20',
    'gym>=0.17.1',
    'pygame>=2.0'
    'svgpath2mpl>=1.0.0'
]

setup(
    name='mobile-env',
    version='0.1.0',
    author='Stefan Schneider',
    description="mobile-env: A minimalist environment for decision making in wireless mobile networks.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/stefanbschneider/mobile-env',
    packages=find_packages(),
    python_requires=">=3.8.*",
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
