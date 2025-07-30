from setuptools import find_packages, setup

setup(
    name='PlantVision', # The name pip uses
    version='1.0.0',
    # find_packages tells setuptools to look for any folders with an __init__.py
    # and treat them as packages. The `where` argument points to the source root.
    packages=find_packages(where='src'),
    package_dir={'': 'src'}, # Tells setuptools that packages are under src
    author='Muhumuza Deus.M.',
    description='A project that leverages a deep-learning vision model to help '
                'farmers and agricultural specialists identify plant diseases '
                'quickly and accurately.',
    license='MIT',
)