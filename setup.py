# python setup.py sdist bdist_wheel
# --twine register dist/*(非必须)
# twine upload dist/*
#
from setuptools import setup, find_packages
import os
def _process_requirements():
    packages = open('requirements.txt').read().strip().split('\n')
    requires = []
    for pkg in packages:
        if pkg.startswith('git+ssh'):
            return_code = os.system('pip install {}'.format(pkg))
            assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
        else:
            requires.append(pkg)
    requires = [i.replace('==','>=') for i in requires]
    return requires


setup(
    name='singletrader',
    version='0.2.0',
    description='a package for backtesting and factor analysis',
    author='Simon X',
    author_email='robortcher@outlook.com',
    packages=find_packages(),
    install_requires=_process_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
