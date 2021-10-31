from setuptools import *

with open('README.md', 'r', encoding='utf-8') as fh:
	long_description= fh.read()


setup(
    name = "xcast",
    version = "0.2.5",
    author = "Kyle Hall",
    author_email = "kjhall@iri.columbia.edu",
    description = ("High Performance Gridpoint-Wise Machine Learning for the Earth Sciences"),
    license = "MIT",
    keywords = "Machine-Learning High-Performance AI Climate Forecasting ",
    url = "https://github.com/kjhall01/xcast/",
    packages=[
		'xcast',
		'xcast.classification',
		'xcast.core',
		'xcast.flat_estimators',
		'xcast.flat_estimators.classifiers',
		'xcast.flat_estimators.regressors',
		'xcast.mme',
		'xcast.preprocessing',
		'xcast.regression',
		'xcast.validation',
		'xcast.verification'],
	package_dir={
		'xcast':'src',
		'xcast.classification':'src/classification',
		'xcast.core':'src/core',
		'xcast.flat_estimators':'src/flat_estimators',
		'xcast.flat_estimators.classifiers':'src/flat_estimators/classifiers',
		'xcast.flat_estimators.regressors':'src/flat_estimators/regressors',
		'xcast.mme':'src/mme',
		'xcast.preprocessing':'src/preprocessing',
		'xcast.regression':'src/regression',
		'xcast.validation':'src/validation',
		'xcast.verification':'src/verification'},
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
