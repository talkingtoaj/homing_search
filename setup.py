import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="homing-search-keras",
    version="0.0.3",
    author="Andrew de Jonge",
    author_email="talkingtoaj@hotmail.com",
    description="Smart hyperparameter optimization in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/talkingtoaj/homing_search",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
)

# To push a new Version
# 1. Update version number above
# 2. $ python setup.py sdist bdist_wheel
# 3. $ python -m twine upload dist/*
