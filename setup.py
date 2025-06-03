from setuptools import setup, find_packages

setup(
    name='KineticFitToolkit',
    version='0.1',
    description='Kinetic fitting toolkit for reaction kinetics',
    authors='Claire Muzyka (Main Developer)' and 'Jean-Christophe M. Monbaliu (Principal Investigator)',
    author_email='cmuzyka@uliege.be',
    packages=find_packages(),
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'kinetic-fit = main:main'
        ]
    },
    include_package_data=True,
    package_data={
        'KineticFitToolkit': ['data/*.csv']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
