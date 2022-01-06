import setuptools


with open("requirements.txt", "r") as fp:
    required = fp.read().splitlines()

setuptools.setup(
    name="online-neural-cdes",
    version="0.0.3",
    author="James Morrill",
    author_email="james.morrill.6@gmail.com",
    description="Online neural CDEs paper",
    url="https://github.com/jambo6/online-neural-cdes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=required,
    extras_require={
        "test": ["pytest"]
    }
)
