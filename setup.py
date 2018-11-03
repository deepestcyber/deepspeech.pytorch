import os

from setuptools import setup, find_packages


version = '0.1.1'

install_requires = [
]

tests_require = [
]

docs_require = [
]

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

try:
    CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    CHANGES = ''

setup(
    name='deepspeech',
    version=version,
    description='DeepSpeech in PyTorch',
    long_description=README,
    license='new BSD 3-Clause',
    packages=find_packages(),
    include_package_data=True,
    url="nope",
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require,
    },
)
