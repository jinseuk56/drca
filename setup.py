from setuptools import setup, find_packages

setup(
    name='drca',
    version='0.7.0',
    description='DR assisted cluster analysis for hyperspectral datasets',
    author='Jinseok Ryu',
    author_email='jinseuk56@gmail.com',
    url='https://github.com/jinseuk56',
    packages=find_packages(include=['drca']),
    install_requires=['numpy', 'matplotlib', 'scikit-learn', 'ipywidgets', 'tifffile', 'ipympl', 'jupyterlab'],
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest']
)
