import os
from setuptools import setup, find_packages


# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


requirements = [
    'structlog>=20.2.0',
    'structlog-round>=1.0',
    'shapely==1.7.0',
    'matplotlib==3.2.1',
    'numpy<1.20',
    'gym[atari]>=0.17.1',
    'pandas>=1.0.5',
    'tqdm==4.47.0',
    'joblib==0.16.0',
    'svgpath2mpl'
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
