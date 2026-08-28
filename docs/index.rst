rayTEM Documentation
====================

rayTEM is a ray-tracing electron optics simulator for TEM instruments, living
inside the ``pySEA`` (Python Simulation Experiment Analysis) namespace package.
One column description drives four propagation modes — geometric rays,
beam-envelope covariance, a fixed-grid paraxial wave, and a scaled-Fresnel
wave — and serializes through sea-eco's ``.sea`` format.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   getting_started
   propagation_modes
   operating_the_column
   terminology

.. toctree::
   :maxdepth: 2
   :caption: Example Scripts

   examples

.. toctree::
   :maxdepth: 2
   :caption: AI Tools

   ai_tools/install

.. toctree::
   :maxdepth: 2
   :caption: Into the SEA-weeds (For developers)

   dev_documentation
   wave-optics-sampling

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference

Notes on the ``pySEA`` namespace
--------------------------------

The distribution package is ``raytem``; it installs the import package
``pySEA.rayTEM`` into the shared ``pySEA`` namespace, alongside siblings such
as ``pySEA.sea_eco`` (the data model this repo serializes through) and
``pySEA.sea_sand`` (identifiers). The generated catalog of installed wiki
slices lives at ``pySEA/ai_wiki/ecosystem/index.md`` in any environment with
the ``sea-ecosystem`` dev tooling installed.
