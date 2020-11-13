import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="homing_search",
    version="0.0.1",
    author="Andrew de Jonge",
    author_email="talkingtoaj@hotmail.com",
    description="Smart hyperparameter optimization in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/talkingtoaj/homing_search",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: 3-Clause BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)