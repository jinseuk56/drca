from setuptools import setup, find_packages

setup(
    name='drca',
    version='0.1.9',
    description='DR assisted cluster analysis for hyperspectral datasets',
    author='Jinseok Ryu',
    author_email='jinseuk56@gmail.com',
    url='https://github.com/jinseuk56',
    packages=find_packages(include=['drca']),
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest']
)
