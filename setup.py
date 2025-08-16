from setuptools import setup, find_packages

setup(
    name='sevae',
    version='0.1.0',
    license='MIT',
    description='Structural Equation–VAE (SE-VAE) for interpretable disentangled latents on tabular data (PyTorch)',
    author='Ruiyu Zhang',
    author_email='ruiyuzh@connect.hku.hk',
    url='',  # optional if no homepage
    keywords=['vae', 'disentanglement', 'representation-learning', 'pytorch', 'tabular-data', 'sem'],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.2',
        'numpy>=1.22',
        'scikit-learn>=1.1',
        'tqdm>=4.65',
        'typing-extensions>=4.5',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)