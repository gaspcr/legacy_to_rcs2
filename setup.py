from setuptools import setup, find_packages


setup(
    name='legacy_to_rcs2',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'astropy',
        'photutils',
        'reproject',
        'scipy',
        'pandas',
        'tqdm',
        'requests',
        'astro-datalab',
    ],
    scripts=[
        'bin/query_degrade_hsc',
        'bin/query_hsc',
        'bin/read_degrade_hsc',
        'bin/hsc_query_example',
    ],
    include_package_data=True,
    author='Rodrigo Iugarte (fork of Felipe Urcelay, LSST SLSC)',
)
