from setuptools import setup

setup(name='hetmech',
      description='A search engine for hetnets',
      long_description='Matrix implementations of path-count-based measures',
      url='https://github.com/greenelab/hetmech',
      license='BSD 3-Clause License',
      packages=['hetmech'],
      install_requires=[
          'hetio>=0.2.8',
          'numpy',
          'scipy',
          'xarray',
          ]
      )
