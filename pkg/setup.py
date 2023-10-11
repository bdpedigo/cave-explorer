from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    "matplotlib",
    "numpy>=1.19",
    "pandas>=2.0",
    "scipy>=1.11.0",
    "seaborn>=0.13.0",
]

setup(
    name="pkg",
    packages=find_packages(),
    version="0.1.0",
    description="Local package",
    author="Ben Pedigo",
    license="MIT",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=[],
)
