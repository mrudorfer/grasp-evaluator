import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

python_versions = '>=3.6, <3.9'  # probably newer versions work as well

requirements_default = [
    'numpy',  # for all datastructures
    'tqdm',  # progress bars
    'matplotlib',  # precision coverage plots
    'attrdict',
    'GPNet-simulator @ git+ssh://git@github.com/mrudorfer/GPNet-simulator'
]

setuptools.setup(
    name='GPNet-evaluator',
    version='0.1',
    python_requires=python_versions,
    install_requires=requirements_default,
    packages=setuptools.find_packages(),
    url='',
    license='',
    author='Martin Rudorfer',
    author_email='m.rudorfer@bham.ac.uk',
    description='evaluation on GPNet dataset',
    long_description=long_description
)
