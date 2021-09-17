from setuptools import *

with open('README.md', 'r', encoding='utf-8') as fh:
	long_description= fh.read()


setup(
    name = "xcast",
    version = "0.1.0",
    author = "Kyle Hall",
    author_email = "kjhall@iri.columbia.edu",
    description = ("High Performance Gridpoint-Wise Machine Learning for the Earth Sciences"),
    license = "MIT",
    keywords = "Machine-Learning High-Performance AI Climate Forecasting ",
    url = "https://github.com/kjhall01/xcast/",
    packages=['xcast', 'xcast.core', 'xcast.mme', 'xcast.preprocessing', 'xcast.validation', 'xcast.verification'],
	package_dir={'xcast':'src', 'xcast.core':'src/core', 'xcast.mme':'src/mme', 'xcast.preprocessing':'src/preprocessing', 'xcast.validation':'src/validation', 'xcast.verification':'src/verification'},
	python_requires=">=3.4",
    long_description=long_description,
	long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
    ],
)
