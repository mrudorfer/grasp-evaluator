import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

python_versions = '>=3.6, <3.9'  # probably newer versions work as well

requirements_default = [
    'numpy',  # for all datastructures
    'tqdm',  # progress bars
    'matplotlib',  # precision coverage plots
    'attrdict',
]

setuptools.setup(
    name='grasp-evaluator',
    version='0.1',
    python_requires=python_versions,
    install_requires=requirements_default,
    packages=setuptools.find_packages(),
    url='',
    license='',
    author='Martin Rudorfer',
    author_email='m.rudorfer@bham.ac.uk',
    description='module for rule- and simulation-based evaluation of grasp predictions',
    long_description=long_description
)

try:
    import gpnet_sim
except ImportError:
    print('*****************************************************************************')
    print('Please install GPNet-simulator (https://github.com/mrudorfer/GPNet-simulator)')
    print('*****************************************************************************')

