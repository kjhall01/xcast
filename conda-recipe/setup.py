from setuptools import *
import os

with open('{}/../README.md'.format(os.getenv('RECIPE_DIR')), 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="xcast",
    version="0.6.9",
    author="Kyle Hall",
    author_email="hallkjc01@gmail.com",
    description=(
        "Python Climate Forecasting Toolkit"),
    license="MIT",
    keywords="Machine-Learning High-Performance AI Climate Forecasting ",
    url="https://github.com/kjhall01/xcast/",
    packages=[
        'xcast',
        'xcast.core',
        'xcast.flat_estimators',
        'xcast.estimators',
        'xcast.preprocessing',
        'xcast.validation',
        'xcast.verification',
        'xcast.visualization'],
    package_data={},
    package_dir={
        'xcast': '{}/../src'.format(os.getenv('RECIPE_DIR')),
        'xcast.core': '{}/../src/core'.format(os.getenv('RECIPE_DIR')),
        'xcast.flat_estimators': '{}/../src/flat_estimators'.format(os.getenv('RECIPE_DIR')),
        'xcast.estimators': '{}/../src/estimators'.format(os.getenv('RECIPE_DIR')),
        'xcast.preprocessing': '{}/../src/preprocessing'.format(os.getenv('RECIPE_DIR')),
        'xcast.validation': '{}/../src/validation'.format(os.getenv('RECIPE_DIR')),
        'xcast.verification': '{}/../src/verification'.format(os.getenv('RECIPE_DIR')),
        'xcast.visualization': '{}/../src/visualization'.format(os.getenv('RECIPE_DIR')),
        },
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
