#!/usr/bin/env python
#
# Copyright (C) 2009-2011 University of Edinburgh
#
# This file is part of IMUSim.
#
# IMUSim is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# IMUSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IMUSim.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
depsOK = True

import numpy as np

try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
except ImportError:
    print("Setuptools must be installed - see http://pypi.python.org/pypi/setuptools")

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
    print("Using Cython to compile modules")
except ImportError:
    USE_CYTHON = False
    print("Using C sources for modules")

if USE_CYTHON:
    def c_to_pyx(sources):
        c2p = lambda path: path.strip().rsplit('.c')[0] + '.pyx'
        return [c2p(path) for path in sources]
else:
    def c_to_pyx(sources):
        return sources

natural_neighbour_sources = [
    'datasetgen/maths/natural_neighbour.{}'.format('pyx' if USE_CYTHON else 'c'),
    'datasetgen/maths/natural_neighbour/utils.c',
    'datasetgen/maths/natural_neighbour/delaunay.c',
    'datasetgen/maths/natural_neighbour/natural.c'
]

ext_modules = [
    Extension("datasetgen.maths.quaternions",
              c_to_pyx(['datasetgen/maths/quaternions.c'])),
    Extension("datasetgen.maths.quat_splines",
              c_to_pyx(['datasetgen/maths/quat_splines.c'])),
    Extension("datasetgen.maths.vectors",
              c_to_pyx(['datasetgen/maths/vectors.c'])),
    Extension("datasetgen.maths.natural_neighbour",
              natural_neighbour_sources)
]

if USE_CYTHON:
    ext_modules = cythonize(ext_modules)
    
packages = find_packages()

if depsOK:
    setup(
        name = "datasetgen",
        version = "0.1",
        author = "Hannes Ovrén",
        license = "GPLv3",
        url = "",
        install_requires = ["Cython"],
        packages = packages,
        include_dirs = [np.get_include()],
        ext_modules = ext_modules
    )

