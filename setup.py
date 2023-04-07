from setuptools import setup, find_packages

setup(
    name='singletrader',
    version='0.1.1',
    description='a package for backtesting and factor analysis',
    author='Simon X',
    author_email='robortcher@outlook.com',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)