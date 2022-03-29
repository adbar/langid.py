import re
from pathlib import Path
from setuptools import setup



def get_version(package):
    "Return package version as listed in `__version__` in `init.py`"
    # version = Path(package, '__init__.py').read_text() # Python >= 3.5
    with open(str(Path(package, '__init__.py')), 'r', encoding='utf-8') as filehandle:
        initfile = filehandle.read()
    return re.search('__version__ = [\'"]([^\'"]+)[\'"]', initfile).group(1)


def get_long_description():
    "Return the README"
    with open('README.rst', 'r', encoding='utf-8') as filehandle:
        long_description = filehandle.read()
    #long_description += "\n\n"
    #with open("CHANGELOG.md", encoding="utf8") as f:
    #    long_description += f.read()
    return long_description


setup(name='py3langid',
    version=get_version('py3langid'),
    description="Fork of the language identification tool langid.py, featuring a modernized codebase and faster execution times.",
    long_description=get_long_description(),
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords=['language detection', 'language identification', 'langid'],
    author='Adrien Barbaresi',
    author_email='barbaresi@bbaw.de',
    url='https://github.com/adbar/py3langid',
    license='BSD',
    packages=['py3langid'],
    include_package_data=True,
    install_requires=[
        'numpy',
    ],
    package_data={'py3langid': ['data/model.plzma']},
    entry_points= {
      'console_scripts': ['langid = py3langid.langid:main'],
    },
    tests_require=['pytest'],
    zip_safe=False,
)