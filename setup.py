# coding: utf-8
from setuptools import setup, find_packages
from re import search as re_search
from pathlib import Path


## INFOS ##
package     = 'rp2biosensor'
descr       = 'Build Sensing-Enabling Metabolic Pathways from RetroPath2.0 output'
url         = 'https://github.com/brsynth/rp2biosensor'
authors     = 'Thomas Duigou'
corr_author = 'thomas.duigou@inrae.fr'

def get_long_description():
    with open(Path(__file__).resolve().parent / 'README.md') as ifh:
        return ifh.read()

def get_version():
    with open(Path(__file__).resolve().parent / 'CHANGELOG.md') as ifh:
        for line in ifh:
            if line.startswith('##'):
                m = re_search("\[(.+)\]", line)
                if m:
                    return m.group(1)

setup(
    name                          = package,
    version                       = get_version(),
    author                        = authors,
    author_email                  = corr_author,
    description                   = descr,
    long_description              = get_long_description(),
    long_description_content_type = 'text/markdown',
    url                           = url,
    packages                      = find_packages(),
    package_dir                   = {package: package},
    include_package_data          = True,
    test_suite                    = 'pytest',
    license                       = 'MIT',
    classifiers                   = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires               = '>=3.6',
    entry_points={
        'console_scripts': [
            'rp2biosensor=rp2biosensor.__main__:main',
        ],
    },
)
