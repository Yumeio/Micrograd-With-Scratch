import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='micrograd-with-scratch',
    version='0.0.1',
    author='Pham Anh Tuan <Yume0308>',
    author_email="23020704@vnu.edu.vn",
    description='A simple autograd engine',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Yume0308/Micrograd-With-Scratch',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)