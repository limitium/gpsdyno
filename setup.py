# GPSDyno - GPS-based vehicle power calculator
# Copyright (C) 2024 GPSDyno Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

setup(
    name='gpsdyno',
    version='1.0.0',
    description='GPS-based vehicle power calculator',
    author='GPSDyno Contributors',
    license='AGPL-3.0',
    packages=find_packages(),
    py_modules=['config', 'gpsdyno_calculator'],
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pynmea2',
        'pillow',
    ],
    python_requires='>=3.8',
)
