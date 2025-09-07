from setuptools import find_packages, setup

setup(
    name='plantvision',
    version='2.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    author='Muhumuza Deus.M.',
    description='An end-to-end deep learning system for detecting '
                'and classifying different kinds of plant diseases '
                'based on their leaves ',
    python_requires='>=3.8',

    entry_points={
        'console_scripts': [
            'plantvision-train = plantvision.train:main',
            'plantvision-evaluate = plantvision.evaluate:main',
            'plantvision-predict = plantvision.predict:main',
            'plantvision-export = plantvision.export:main',
            'plantvision-benchmark = plantvision.benchmark:main'
        ]
    },

    license='MIT',
)