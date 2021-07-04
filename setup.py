import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MatterIx",
    version="1.1.1",
    author="Siddesh Sambasivam Suseela",
    author_email="plutocrat45@gmail.com",
    description="Just another deep learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SiddeshSambasivam/MatterIx",
    project_urls={
        "Bug Tracker": "https://github.com/SiddeshSambasivam/MatterIx/issues",
    },
    install_requires=[
        "appdirs==1.4.4",
        "attrs==21.2.0",
        "black==21.6b0",
        "certifi==2021.5.30",
        "click==8.0.1",
        "iniconfig==1.1.1",
        "mypy-extensions==0.4.3",
        "numpy==1.20.3",
        "packaging==20.9",
        "pathspec==0.8.1",
        "pluggy==0.13.1",
        "py==1.10.0",
        "pyparsing==2.4.7",
        "pytest==6.2.4",
        "regex==2021.4.4",
        "toml==0.10.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8.0",
)
