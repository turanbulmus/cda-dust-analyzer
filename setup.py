import setuptools

setuptools.setup(
    name='cda_dust_agent',
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'numpy',
        'scipy',
        'pyarrow',
        'google-cloud-storage'
    ]
)
