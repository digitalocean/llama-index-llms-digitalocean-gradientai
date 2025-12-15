"""Setup script for llama-index-llms-gradient."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llama-index-llms-digitalocean-gradientai",
    version="0.1.1",
    author="Narasimha Badrinath",
    author_email="bnarasimha21@gmail.com",
    description="LlamaIndex integration for DigitalOcean Gradient AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bnarasimha21/llamaindex-digitalocean-gradientai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "llama-index-core>=0.10.0",
        "gradient>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
)
