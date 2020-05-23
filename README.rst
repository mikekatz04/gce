########################################
gce: GPU-Accelerated Condtional Entropy
########################################

gce implements the conditional entropy search technique

.. _arXiv:1807.02511: https://arxiv.org/abs/1807.02511
.. _arXiv:1508.07250: https://arxiv.org/abs/1508.07250
.. _arXiv:1508.07253: https://arxiv.org/abs/1508.07253

The three plots to choose from are Waterfall, Ratio, and Horizon. A Waterfall plot is a filled contour plot similar to figure 3 in the LISA Mission Proposal (arxiv:1702.00786). Ratio plots show the ratio of SNRs between two different binary and sensitivity configurations. They also include loss/gain contours showing where two configurations differ in terms of the sources they can and cannot detect. Horizon plots show line contours of multiple configurations for a given SNR value. See the original paper and notebook examples for more information.

**Note**: The remainder of this introduction specifically details installation of the plotting module. It is very adaptable to different gravitational wave sources or other measurements with similar signal-to-noise properties. The ``snr_calculator`` package is listed as a requirement for BOWIE (meaning the plotting module). This package is installed with the bowie install, however, it is available separately from the plotting module. For this purpose, it has its own README and documentation `here`_.

.. _here: https://mikekatz04.github.io/BOWIE/snrcalc_link.html

Getting Started
===============

These instructions will get you a copy of the project up and running on your local machine for usage and testing purposes.

Available via pip and on github: https://github.com/mikekatz04/BOWIE

Prerequisites
=============

It is best to run out of conda environment. It will handle the dependencies better. If you have issues with certain modules, try to update them.

Software installation/usage only requires a few specific libraries in python. If you install with pip, all of these libraries should be automatically installed if you do not have them (this includes ``snr_calculator``, which is required). All python libraries are included with Anaconda. If you do not run python in an anaconda environment, you  will need the following libraries and modules to run with all capabilities: Numpy, Scipy, collections, sys, json, multiprocessing, datetime, time, astropy, h5py, and matplotlib. All can be installed with pip. For example, within your python environment of choice:

``pip install h5py``

In order to properly create waveforms with ctypes, you will need complex, gsl, and math c libraries. For installing gsl, refer to https://www.gnu.org/software/gsl/ or install it through anaconda.

``gwsnrcalc`` is also required. This will install automatically with pip install or setup.py.


Installation
=============

Begin with updating conda:
  ``conda update conda``

Create a conda environment (change the name as desired. Default: bowie_env):
  ``conda create -n bowie_env numpy scipy astropy h5py gsl matplotlib jupyter ipython python=3.7``

Installation is done two ways:

1) using pip

  ``pip install bowie``

  This will download all necessary packages to your current environment. It will not download the notebooks for testing and example usage.

2) Clone the git repo on the command line, or download it from github. This is for all the modules, example jupyter notebooks, and extra files.

  a) navigate to the directory of your choice.

  b) clone the git repo on the command line.

    ``git clone https://github.com/mikekatz04/BOWIE.git``

  c) pip install the local files to add the modules to your environment and compile the c codes.

    ``pip install ./BOWIE/``


Testing and Running an Example
==============================

To test the codes, you run the testing notebook.

``jupyter notebook quick_testing_example.ipynb``

Contributing
============

Please read `CONTRIBUTING.md`_ for details on our code of conduct, and the process for submitting pull requests to us.

.. _CONTRIBUTING.md: https://gist.github.com/PurpleBooth/b24679402957c63ec426

Versioning
=============

Current version is 2.0.1.

We use `SemVer`_ for versioning.

.. _SemVer: http://semver.org/

Authors
=======

* **Michael Katz** - `mikekatz04`_

.. _mikekatz04: https://github.com/mikekatz04/

Please email the author with any bugs or requests.

License
=======

This project is licensed under the GNU License - see the `LICENSE.md`_ file for details.

.. _LICENSE.md: https://github.com/mikekatz04/BOWIE/blob/master/LICENSE

Acknowledgments
===============

* Thanks to Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London for authorship of the original c code for PhenomD within LALsuite.
