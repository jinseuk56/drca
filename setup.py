from setuptools import setup, find_packages

setup(
    name='drca',
    version='1.0.1',
    description='DR assisted cluster analysis for hyperspectral datasets',
    author='Jinseok Ryu',
    author_email='jinseuk56@gmail.com',
    url='https://github.com/jinseuk56',
    packages=find_packages(include=['drca']),
    install_requires=[
        'numpy', 
        'matplotlib', 
        'scikit-learn', 
        'ipywidgets', 
        'tifffile', 
        'ipympl', 
        'jupyterlab',
        'streamlit'  # Added Streamlit dependency
    ],
    entry_points={
        'console_scripts': [
            'drca-gui=drca.cli:run_gui',  # Creates the terminal command
        ],
    },
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest']
)