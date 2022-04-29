====================================
GoSafeOpt - Safe Bayesian Optimization
====================================


This repository is the official implementation accompanying the paper "GoSafeOpt: Scalable Safe Exploration for Global Optimization of Dynamical Systems" by Bhavya Sukhija, Matteo Turchetta, David Lindner, Andreas Krause, Sebastian Trimpe, and Dominik Baumann. It builds upon the SafeOpt code provided at https://github.com/befelix/SafeOpt. GoSafeOpt is an extension of [1]_ based on particle swarms, as proposed by [2]_, which can be applied for general high dimensional tasks.


.. [1] D.Baumann, A.Marco, M.Turchetta, S.Trimpe,
  `GoSafe Globally optimal safe robot learning <https://arxiv.org/abs/2105.13281>`_,
  in IEEE International Conference on Roboticsand Automation (ICRA), 2021.

.. [2] Rikky R.P.R. Duivenvoorden, Felix Berkenkamp, Nicolas Carion, Andreas Krause, Angela P. Schoellig,
  `Constrained Bayesian optimization with Particle Swarms for Safe Adaptive Controller Tuning <http://www.dynsyslab.org/wp-content/papercite-data/pdf/duivenvoorden-ifac17.pdf>`_,
  in Proc. of the IFAC (International Federation of Automatic Control) World Congress, 2017.

  .. [1] B.Sukhija, M.Turchetta, D.Lindner, A.Krause, D.Baumann, S.Trimpe,
  `GoSafeOpt Scalable Safe Exploration for Global Optimization of Dynamical Systems <https://arxiv.org/abs/2201.09562>`_,

Contributions:
---------------
SafeOpt, SafeOptSwarm developed by Felix Berkenkamp

GoSafeOpt developed by Bhavya Sukhija

Installation
------------
You can clone the repository and install it using

``pip install .``


License
-------

The code is licenced under the MIT license and free to use by anyone without any restrictions.


Reproducing Results
-------------------

To reproduce our simulation results, the repository: https://github.com/Data-Science-in-Mechanical-Engineering/franka-emika-panda-simulation is required.

An example of GoSafeOpt on a 1D toy example is also provided in the examples folder.
