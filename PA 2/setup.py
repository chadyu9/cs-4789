from setuptools import setup, find_packages

setup(
    name="cs4789-pa2",
    version="0.0.1",
    packages=["pa2"],
    python_requires=">=3.10",
    install_requires=[
        "gym==0.7.4",
        "numpy==1.26.3",
        "tqdm==4.66.1",
        "pyglet==1.5.27",
        "autograd"
    ],
)