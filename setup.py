import pathlib
import re

import setuptools

directory = pathlib.Path(__file__).parent.resolve()

# version
init_path = directory.joinpath("hetmatpy", "__init__.py")
text = init_path.read_text(encoding="utf-8-sig")
pattern = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.MULTILINE)
version = pattern.search(text).group(1)

# long_description
readme_path = directory.joinpath("README.md")
long_description = readme_path.read_text(encoding="utf-8-sig")

setuptools.setup(
    name='hetmatpy',
    description='Matrix implementations for hetnets and path-count-based measures',
    long_description_content_type="text/markdown",
    long_description=long_description,
    url='https://github.com/hetio/hetmatpy',
    project_urls={
        "Source": "https://github.com/hetio/hetmatpy",
        "Documentation": "https://hetio.github.io/hetmatpy",
        "Tracker": "https://github.com/hetio/hetmatpy/issues",
        "Homepage": "https://het.io/software/",
        "Publication": "https://greenelab.github.io/connectivity-search-manuscript/",
    },
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
