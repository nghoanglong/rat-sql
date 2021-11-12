from setuptools import setup, find_packages

setup(
    name='RAT-SQL',
    version='1.0',
    description='A relation-aware semantic parsing model from English to SQL',
    packages=find_packages(exclude=["*_test.py", "test_*.py"]),
    package_data={
        '': ['*.asdl'],
    },
    install_requires=[
        'asdl~=0.1.5',
        'astor~=0.8.1',
        'attrs~=18.2.0',
        'babel~=2.7.0',
        'bpemb~=0.2.12',
        'cython~=0.29.24',
        'entmax~=1.0',
        'jsonnet~=0.14.0',
        'networkx~=2.6.3',
        'nltk~=3.6.5',
        'numpy~=1.19.5',
        'pyrsistent~=0.14.11',
        'pytest~=5.3.5',
        'records~=0.5.3',
        'stanford-corenlp~=3.9.2',
        'tabulate~=0.8.9',
        'tqdm>=4.62.3',
        'transformers~=4.12.3',
        'stanza~=1.3.0',
        'vncorenlp~=1.0.3',
    ],
    entry_points={"console_scripts": ["ratsql=run:main"]},
)
