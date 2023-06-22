import setuptools

setuptools.setup(
    name="vqtorch",
    packages=setuptools.find_packages(),
    version="0.1.0",
    author="Minyoung Huh",
    author_email="minhuh@mit.edu",
    description=f"vector-quantization for pytorch",
    url="git@github.com:minyoungg/vqtorch.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
            "torch>=1.13.0",
            "string-color==1.2.3",
            "torchpq==0.3.0.1",
    ],
    python_requires='>=3.6', # developed on 3.9 / 3.10
)
