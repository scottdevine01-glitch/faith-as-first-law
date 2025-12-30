from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="faith-as-first-law",
    version="0.1.0",
    author="Scott Devine",
    author_email="scottdevine01@gmail.com",
    description="Experimental protocols for testing the Faith prior in physics, morality, and materials science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scottdevine/faith-as-first-law",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "heartpy>=1.2.0",
            "neurokit2>=0.2.0",
            "networkx>=3.0",
            "torch>=2.0.0",
            "gymnasium>=0.28.0",
            "stable-baselines3>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "faith-experiments=faith_as_first_law.cli:main",
        ],
    },
)
