from setuptools import find_packages
from setuptools import setup

install_requires = [
    'torch>=1.5.1',
    'gym>=0.17.2',
    'numpy>=1.10.4',
    'pillow',
]

setup(
    name='pfrlx',
    version='0.0.1',
    description='',
    author='Hiroki Furuta',
    author_email='',
    url='',
    license='MIT License',
    packages=find_packages(),
    install_requires=install_requires,
)
