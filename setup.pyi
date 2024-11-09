import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

NAME = "micrograd-with-scratch" 
VERSION = "0.0.1"
DESCRIPTION = "A simple autograd engine"
AUTHOR = "Pham Anh Tuan <Yumeio>"
AUTHOR_EMAIL = "23020704@vnu.edu.vn"
    
setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Yumeio/Micrograd-With-Scratch',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)