from setuptools import setup

setup(
    name = 'servier',
    packages = ['servier'],
    entry_points = {
            "console_scripts": ['servier = servier:main']
        },
    version = '0.0.1',
    author = 'Marc Duda',
    author_email = 'marc.duda@free.fr',
    install_requires=[
        #'numpy==1.19.2',
        #'scipy==1.5.2',
        'scikit-learn',
        'pyparsing',
        'matplotlib<3.4',
        'pandas',
        'imbalanced-learn',
        'tensorflow==2.4.0',
        'flask',
        'flask_cors',
        'click'
    ],
)
