from setuptools import setup, find_packages

setup(
    name='kluppspy',
    version='0.1.1',
    description='A Package of useful Computer Vision constructs and functions',
    url='https://github.com//klupp/kluppspy',
    author='Aleksandar Kuzmanoski',
    author_email='aleksandar.kuzmanoski@rwth-aachen.de',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'imageio',
        'scipy',
        'matplotlib',
#         'opencv'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==6.2.5'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
