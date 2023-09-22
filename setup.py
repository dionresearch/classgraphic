#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = ["plotly>=5.0", "scikit-learn", "pandas", "numpy", "nbformat"]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3.5"]

setup(
    author="Francois Dion",
    author_email="fdion@dionresearch.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="Interactive classification diagnostic plots",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords=["classgraphic", "classification", "clustering", "visualization", "ml", "machine learning", "plotly", "interactive"],
    name="classgraphic",
    packages=find_packages(include=["classgraphic", "classgraphic.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/dionresearch/classgraphic",
    version="0.3.1",
    zip_safe=False,
)
