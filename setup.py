from __future__ import print_function
from setuptools import setup


setup(name='rayTEM',
      version='1.0',
      description='Electron optics simulator for a transmission electron microscope.',
      author='Eric Hoglund, Andy Lupini',
      author_email='hoglunder@ornl.gov',
      packages=['rayTEM'],
      install_requires=['numpy', 'matplotlib', 'pandas']
     )