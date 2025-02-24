from setuptools import setup, find_packages

setup(
    name='Annalisa',
    version='2.0',  # Update with your package version
    packages=find_packages(),
    #install_requires=[  # Add any dependencies your package requires
    #    'dependency1',
    #    'dependency2',
    #],
    # Additional metadata
    author=['Lorenzo Asprea','Francesco Sarandrea'],
    author_email=['asprea@to.infn.it', 'francesco.sarandrea@to.infn.it'],
    description='A package used to find correlations among the glitch signals in different auxiliary channels and the strain channel in Virgo.',
    install_requires=[
            "tqdm",
            "Pillow",  # PIL is installed as Pillow
            "matplotlib",
            "numpy",
            "gwpy",
            "pandas",
            "h5py",
            "torch",
            "ml4gw"]
)
