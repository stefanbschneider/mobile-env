import os
from setuptools import setup, find_packages


# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


requirements = [

]

eval_requirements = [

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
    install_requires=requirements + eval_requirements,
    zip_safe=False,
    # entry_points={
        # 'console_scripts': [
            # 'deepcomp=deepcomp.main:main'
        # ]
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
