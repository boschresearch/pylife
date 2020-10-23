import setuptools, sys, os
import distutils

use_cython = False

if sys.platform == 'win32':
    try:
        msvc = setuptools.msvc.EnvironmentInfo('win32')
        use_cython = True
    except distutils.errors.DistutilsPlatformError:
        pass
else:
    use_cython = True

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    def cythonize (*args, ** kwargs ):
        from Cython.Build import cythonize
        return cythonize(*args, ** kwargs)
    use_cython = False

import versioneer
version=versioneer.get_version()
cmdclass=versioneer.get_cmdclass()

if use_cython:
    cmdclass.update(build_ext=build_ext)

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

cython_modules = [
    setuptools.Extension('pylife.stress.rainflow', sources=['pylife/stress/rainflow.py'])
]

if os.getenv('NO_INSTALL_REQUIREMENTS') != 'true':
    with open("requirements.txt", 'r') as fh:
        requirements = fh.readlines()
else:
    requirements = []

setuptools.setup(
    name="pylife",
    version=version,
    cmdclass=cmdclass,
    author="pyLife developer team @ Bosch Research",
    author_email="Johannes.Mueller4@de.bosch.com",
    description="General Fatigue library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boschresearch/pylife",
    packages=setuptools.find_packages(exclude=['demos', 'tests.*', 'tests']),
    ext_modules = cythonize(cython_modules, compiler_directives={'linetrace': True}) if use_cython else cython_modules,
    include_package_data=True,
	classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
    ],
    setup_requires = [
        'setuptools',
        'cython'
    ],
    install_requires = [ requirements ]
)
