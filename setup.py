from setuptools import setup

setup(
    name='warprigid',
    version='0.0.0',
    packages=['warprigid'],
    package_dir={'':'src'},
    install_requires=[
        'numpy',
        'meshio',
        'scipy',
    ],
)
