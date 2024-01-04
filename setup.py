from setuptools import setup, find_packages

setup(
    name="stats_utils",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "statsmodels",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    author="Toby Wise",
    description="Utilities for statistical analysis",
    url="https://github.com/the-wise-lab/stats-utils",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

