====================================
Contextual GoSafe - Safe Bayesian Optimization
====================================



This code is build upon the SafeOpt repository from https://github.com/befelix/SafeOpt.
It implements Contextual GoSafe, an extension of [1]_ based on particle swarms, as proposed by [2]_, which can be applied for general high dimensional tasks.


.. [1] D.Baumann, A.Marco, M.Turchetta, S.Trimpe, 
`GoSafe: Globally op-timal safe robot learning <https://arxiv.org/abs/2105.13281>`_, 
in IEEE International Conference on Roboticsand Automation (ICRA), 2021.

.. [2] Rikky R.P.R. Duivenvoorden, Felix Berkenkamp, Nicolas Carion, Andreas Krause, Angela P. Schoellig,
  `Constrained Bayesian optimization with Particle Swarms for Safe Adaptive Controller Tuning <http://www.dynsyslab.org/wp-content/papercite-data/pdf/duivenvoorden-ifac17.pdf>`_,
  in Proc. of the IFAC (International Federation of Automatic Control) World Congress, 2017.


Warning: Maintenance mode
-------------------------
This package is no longer actively maintained. That bein said, pull requests to add functionality or fix bugs are always welcome.

Installation
------------
The easiest way to install the necessary python libraries is by installing pip (e.g. ``apt-get install python-pip`` on Ubuntu) and running

``pip install safeopt``

Alternatively you can clone the repository and install it using

``python setup.py install``


License
-------

The code is licenced under the MIT license and free to use by anyone without any restrictions.
