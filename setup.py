from setuptools import setup, find_packages

setup(
    name='dear',
    version='1.0.0',
    description='DEAR: Dissect and Prune for robust AI-generated image detection',
    url='https://github.com/dahyedahye/dear',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    python_requires='>=3.10',
)
