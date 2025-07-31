from setuptools import find_packages, setup

setup(
    name='PlantVision', # The name pip uses
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'}, # Tells setuptools that packages are under src
    author='Muhumuza Deus.M.',
    description='An end-to-end deep learning system for detecting '
                'and classifying different kinds of plant diseases '
                'based on their leaves ',

    license='MIT',
)