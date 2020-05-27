import setuptools

setuptools.setup(
    name='hetmatpy',
    description='A search engine for hetnets',
    long_description='Matrix implementations of path-count-based measures',
    url='https://github.com/hetio/hetmatpy',
    license='BSD-2-Clause Plus Patent License',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'hetnetpy>=0.3.0',
        'numpy',
        'pandas',
        'scipy',
    ],
    extras_require={
        'dev': [
            'black',
            'flake8',
            'portray',
            'pytest',
            'xarray',
        ]
    }
)
