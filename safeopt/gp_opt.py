"""
Classes that implement SafeOpt.

Authors: - Felix Berkenkamp (befelix at inf dot ethz dot ch)
         - Nicolas Carion (carion dot nicolas at gmail dot com)
"""

from __future__ import print_function, absolute_import, division

from collections import Sequence
from functools import partial

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from scipy.stats import norm
from builtins import range

from .utilities import (plot_2d_gp, plot_3d_gp, plot_contour_gp,
                        linearly_spaced_combinations)
from .swarm import SwarmOptimization


import logging


__all__ = ['SafeOpt', 'SafeOptSwarm',"GoSafeSwarm_Contextual"]


class GaussianProcessOptimization(object):
    """
    Base class for GP optimization.

    Handles common functionality.

    Parameters
    ----------
    gp: GPy Gaussian process
    fmin : float or list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted).
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
        If a list, this represents the stopping criterion for all the gps.
        This ignores the scaling factor.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of
        each kernel. You should probably leave this to "auto" unless your
        kernel is non-stationary.
    """

    def __init__(self, gp, fmin, beta=2, num_contexts=0, threshold=0,
                 scaling='auto'):
        """Initialization, see `GaussianProcessOptimization`."""
        super(GaussianProcessOptimization, self).__init__()

        if isinstance(gp, list):
            self.gps = gp
        else:
            self.gps = [gp]
        self.gp = self.gps[0]

        self.fmin = fmin
        if not isinstance(self.fmin, list):
            self.fmin = [self.fmin] * len(self.gps)
        self.fmin = np.atleast_1d(np.asarray(self.fmin).squeeze())

        if hasattr(beta, '__call__'):
            # Beta is a function of t
            self.beta = beta
        else:
            # Assume that beta is a constant
            self.beta = lambda t: beta

        if scaling == 'auto':
            dummy_point = np.zeros((1, self.gps[0].input_dim))
            self.scaling = [gpm.kern.Kdiag(dummy_point)[0] for gpm in self.gps]
            self.scaling = np.sqrt(np.asarray(self.scaling))
        else:
            self.scaling = np.asarray(scaling)
            if self.scaling.shape[0] != len(self.gps):
                raise ValueError("The number of scaling values should be "
                                 "equal to the number of GPs")

        self.threshold = threshold
        self._parameter_set = None
        self.bounds = None
        self.num_samples = 0
        self.num_contexts = num_contexts

        self._x = None
        self._y = None
        self._get_initial_xy()

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def data(self):
        """Return the data within the GP models."""
        return self._x, self._y

    @property
    def t(self):
        """Return the time step (number of measurements)."""
        return self._x.shape[0]

    def _get_initial_xy(self):
        """Get the initial x/y data from the GPs."""
        self._x = self.gp.X
        y = [self.gp.Y]

        for gp in self.gps[1:]:
            if np.allclose(self._x, gp.X):
                y.append(gp.Y)
            else:
                raise NotImplemented('The GPs have different measurements.')

        self._y = np.concatenate(y, axis=1)

    def plot(self, n_samples, axis=None, figure=None, plot_3d=False,
             **kwargs):
        """
        Plot the current state of the optimization.

        Parameters
        ----------
        n_samples: int
            How many samples to use for plotting
        axis: matplotlib axis
            The axis on which to draw (does not get cleared first)
        figure: matplotlib figure
            Ignored if axis is already defined
        plot_3d: boolean
            If set to true shows a 3D plot for 2 dimensional data
        """
        # Fix contexts to their current values
        if self.num_contexts > 0 and 'fixed_inputs' not in kwargs:
            kwargs.update(fixed_inputs=self.context_fixed_inputs)

        true_input_dim = self.gp.kern.input_dim - self.num_contexts
        if true_input_dim == 1 or plot_3d:
            inputs = np.zeros((n_samples ** true_input_dim, self.gp.input_dim))
            inputs[:, :true_input_dim] = linearly_spaced_combinations(
                self.bounds[:true_input_dim],
                n_samples)

        if not isinstance(n_samples, Sequence):
            n_samples = [n_samples] * len(self.bounds)

        axes = []
        if self.gp.input_dim - self.num_contexts == 1:
            # 2D plots with uncertainty
            for gp, fmin in zip(self.gps, self.fmin):
                if fmin == -np.inf:
                    fmin = None
                ax = plot_2d_gp(gp, inputs, figure=figure, axis=axis,
                                fmin=fmin, **kwargs)
                axes.append(ax)
        else:
            if plot_3d:
                for gp in self.gps:
                    plot_3d_gp(gp, inputs, figure=figure, axis=axis, **kwargs)
            else:
                for gp in self.gps:
                    plot_contour_gp(gp,
                                    [np.linspace(self.bounds[0][0],
                                                 self.bounds[0][1],
                                                 n_samples[0]),
                                     np.linspace(self.bounds[1][0],
                                                 self.bounds[1][1],
                                                 n_samples[1])],
                                    figure=figure,
                                    axis=axis)

    def _add_context(self, x, context):
        """Add the context to a vector.

        Parameters
        ----------
        x : ndarray
        context : ndarray

        Returns
        -------
        x_extended : ndarray
        """
        context = np.atleast_2d(context)
        num_contexts = context.shape[1]

        x2 = np.empty((x.shape[0], x.shape[1] + num_contexts), dtype=float)
        x2[:, :x.shape[1]] = x
        x2[:, x.shape[1]:] = context
        return x2

    def _add_data_point(self, gp, x, y, context=None):
        """Add a data point to a particular GP.

        This should only be called on its own if you know what you're doing.
        This does not update the global data stores self.x and self.y.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        context: array_like
            The context(s) used for the data points
        gp: instance of GPy.model.GPRegression
            If specified, determines the GP to which we add the data point
            to. Note that this should only be used if that data point is going
            to be removed again.
        """
        if context is not None:
            x = self._add_context(x, context)

        gp.set_XY(np.vstack([gp.X, x]),
                  np.vstack([gp.Y, y]))

    def add_new_data_point(self, x, y, context=None):
        """
        Add a new function observation to the GPs.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        context: array_like
            The context(s) used for the data points.
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        if self.num_contexts:
            x = self._add_context(x, context)

        for i, gp in enumerate(self.gps):
            not_nan = ~np.isnan(y[:, i])
            if np.any(not_nan):
                # Add data to GP (context already included in x)
                self._add_data_point(gp, x[not_nan, :], y[not_nan, [i]])

        # Update global data stores
        self._x = np.concatenate((self._x, x), axis=0)
        self._y = np.concatenate((self._y, y), axis=0)

    def _remove_last_data_point(self, gp):
        """Remove the last data point of a specific GP.

        This does not update global data stores, self.x and self.y.

        Parameters
        ----------
            gp: Instance of GPy.models.GPRegression
                The gp that the last data point should be removed from
        """
        gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])

    def remove_last_data_point(self):
        """Remove the data point that was last added to the GP."""
        last_y = self._y[-1]

        for gp, yi in zip(self.gps, last_y):
            if not np.isnan(yi):
                gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])

        self._x = self._x[:-1, :]
        self._y = self._y[:-1, :]


class SafeOpt(GaussianProcessOptimization):
    """A class for Safe Bayesian Optimization.

    This class implements the `SafeOpt` algorithm. It uses a Gaussian
    process model in order to determine parameter combinations that are safe
    with high probability. Based on these, it aims to both expand the set of
    safe parameters and to find the optimal parameters within the safe set.

    Parameters
    ----------
    gp: GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints.
    parameter_set: 2d-array
        List of parameters
    fmin: list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted)
    lipschitz: list of floats
        The Lipschitz constant of the system, if None the GP confidence
        intervals are used directly.
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
        If a list, this represents the stopping criterion for all the gps.
        This ignores the scaling factor.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of
        each kernel. You should probably leave this to "auto" unless your
        kernel is non-stationary.

    Examples
    --------
    >>> from safeopt import SafeOpt
    >>> from safeopt import linearly_spaced_combinations
    >>> import GPy
    >>> import numpy as np

    Define a Gaussian process prior over the performance

    >>> x = np.array([[0.]])
    >>> y = np.array([[1.]])
    >>> gp = GPy.models.GPRegression(x, y, noise_var=0.01**2)

    >>> bounds = [[-1., 1.]]
    >>> parameter_set = linearly_spaced_combinations([[-1., 1.]],
    ...                                              num_samples=100)

    Initialize the Bayesian optimization and get new parameters to evaluate

    >>> opt = SafeOpt(gp, parameter_set, fmin=[0.])
    >>> next_parameters = opt.optimize()

    Add a new data point with the parameters and the performance to the GP. The
    performance has normally be determined through an external function call.

    >>> performance = np.array([[1.]])
    >>> opt.add_new_data_point(next_parameters, performance)
    """

    def __init__(self, gp, parameter_set, fmin, lipschitz=None, beta=2,
                 num_contexts=0, threshold=0, scaling='auto'):
        """Initialization, see `SafeOpt`."""
        super(SafeOpt, self).__init__(gp,
                                      fmin=fmin,
                                      beta=beta,
                                      num_contexts=num_contexts,
                                      threshold=threshold,
                                      scaling=scaling)

        if self.num_contexts > 0:
            context_shape = (parameter_set.shape[0], self.num_contexts)
            self.inputs = np.hstack((parameter_set,
                                     np.zeros(context_shape,
                                              dtype=parameter_set.dtype)))
            self.parameter_set = self.inputs[:, :-self.num_contexts]
        else:
            self.inputs = self.parameter_set = parameter_set

        self.liptschitz = lipschitz

        if self.liptschitz is not None:
            if not isinstance(self.liptschitz, list):
                self.liptschitz = [self.liptschitz] * len(self.gps)
            self.liptschitz = np.atleast_1d(
                np.asarray(self.liptschitz).squeeze())

        # Value intervals-initialized as empty Q.shape()  (# num parameters combinations, 2*(1+#num_constraints)
        self.Q = np.empty((self.inputs.shape[0], 2 * len(self.gps)),
                          dtype=np.float)

        # Safe set initialized as 0s
        self.S = np.zeros(self.inputs.shape[0], dtype=np.bool)

        # Switch to use confidence intervals for safety
        if lipschitz is None:
            self._use_lipschitz = False
        else:
            self._use_lipschitz = True

        # Set of expanders and maximizers
        self.G = self.S.copy()
        self.M = self.S.copy()

    @property
    def use_lipschitz(self):
        """
        Boolean that determines whether to use the Lipschitz constant.

        By default this is set to False, which means the adapted SafeOpt
        algorithm is used, that uses the GP confidence intervals directly.
        If set to True, the `self.lipschitz` parameter is used to compute
        the safe and expanders sets.
        """
        return self._use_lipschitz

    @use_lipschitz.setter
    def use_lipschitz(self, value):
        if value and self.liptschitz is None:
            raise ValueError('Lipschitz constant not defined')
        self._use_lipschitz = value

    @property
    def parameter_set(self):
        """Discrete parameter samples for Bayesian optimization."""
        return self._parameter_set

    @parameter_set.setter
    def parameter_set(self, parameter_set):
        self._parameter_set = parameter_set

        # Plotting bounds (min, max value
        self.bounds = list(zip(np.min(self._parameter_set, axis=0),
                               np.max(self._parameter_set, axis=0)))
        self.num_samples = [len(np.unique(self._parameter_set[:, i]))
                            for i in range(self._parameter_set.shape[1])]

    @property
    def context_fixed_inputs(self):
        """Return the fixed inputs for the current context."""
        n = self.gp.input_dim - 1
        nc = self.num_contexts
        if nc > 0:
            contexts = self.inputs[0, -self.num_contexts:]
            return list(zip(range(n, n - nc, -1), contexts))

    @property
    def context(self):
        """Return the current context variables."""
        if self.num_contexts:
            return self.inputs[0, -self.num_contexts:]

    @context.setter
    def context(self, context):
        """Set the current context and update confidence intervals.

        Parameters
        ----------
        context: ndarray
            New context that should be applied to the input parameters
        """
        if self.num_contexts:
            if context is None:
                raise ValueError('Need to provide value for context.')
            self.inputs[:, -self.num_contexts:] = context

    def update_confidence_intervals(self, context=None):
        """Recompute the confidence intervals form the GP.

        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        beta = self.beta(self.t)

        # Update context to current setting
        self.context = context

        # Iterate over all functions
        for i in range(len(self.gps)):
            # Evaluate acquisition function
            mean, var = self.gps[i].predict_noiseless(self.inputs)

            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())

            # Update confidence intervals
            self.Q[:, 2 * i] = mean - beta * std_dev
            self.Q[:, 2 * i + 1] = mean + beta * std_dev

    def compute_safe_set(self):
        """Compute only the safe set based on the current confidence bounds."""
        # Update safe set
        self.S[:] = np.all(self.Q[:, ::2] > self.fmin, axis=1)

    def compute_sets(self, full_sets=False):
        """
        Compute the safe set of points, based on current confidence bounds.

        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        full_sets: boolean
            Whether to compute the full set of expanders or whether to omit
            computations that are not relevant for running SafeOpt
            (This option is only useful for plotting purposes)
        """
        beta = self.beta(self.t)

        # Update safe set
        self.compute_safe_set()

        # Reference to confidence intervals
        l, u = self.Q[:, :2].T

        if not np.any(self.S):
            self.M[:] = False
            self.G[:] = False
            return

        # Set of possible maximisers
        # Maximizers: safe upper bound above best, safe lower bound
        self.M[:] = False
        self.M[self.S] = u[self.S] >= np.max(l[self.S])
        max_var = np.max(u[self.M] - l[self.M]) / self.scaling[0]

        # Optimistic set of possible expanders
        l = self.Q[:, ::2]
        u = self.Q[:, 1::2]

        self.G[:] = False

        # For the run of the algorithm we do not need to calculate the
        # full set of potential expanders:
        # We can skip the ones already in M and ones that have lower
        # variance than the maximum variance in M, max_var or the threshold.
        # Amongst the remaining ones we only need to find the
        # potential expander with maximum variance
        if full_sets:
            s = self.S
        else:
            # skip points in M, they will already be evaluated
            s = np.logical_and(self.S, ~self.M)

            # Remove points with a variance that is too small
            s[s] = (np.max((u[s, :] - l[s, :]) / self.scaling, axis=1) >
                    max_var)
            s[s] = np.any(u[s, :] - l[s, :] > self.threshold * beta, axis=1)

            if not np.any(s):
                # no need to evaluate any points as expanders in G, exit
                return

        def sort_generator(array):
            """Return the sorted array, largest element first."""
            return array.argsort()[::-1]

        # set of safe expanders
        G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

        if not full_sets:
            # Sort, element with largest variance first
            sort_index = sort_generator(np.max(u[s, :] - l[s, :],
                                               axis=1))
        else:
            # Sort index is just an enumeration of all safe states
            sort_index = range(len(G_safe))

        for index in sort_index:
            if self.use_lipschitz:
                # Distance between current index point and all other unsafe
                # points
                d = cdist(self.inputs[s, :][[index], :],
                          self.inputs[~self.S, :])

                # Check if expander for all GPs
                for i in range(len(self.gps)):
                    # Skip evaluation if 'no' safety constraint
                    if self.fmin[i] == -np.inf:
                        continue
                    # Safety: u - L * d >= fmin
                    G_safe[index] =\
                        np.any(u[s, i][index] - self.liptschitz[i] * d >=
                               self.fmin[i])
                    # Stop evaluating if not expander according to one
                    # safety constraint
                    if not G_safe[index]:
                        break
            else:
                # Check if expander for all GPs
                for i, gp in enumerate(self.gps):
                    # Skip evlauation if 'no' safety constraint
                    if self.fmin[i] == -np.inf:
                        continue

                    # Add safe point with its max possible value to the gp
                    self._add_data_point(gp=gp,
                                         x=self.parameter_set[s, :][index, :],
                                         y=u[s, i][index],
                                         context=self.context)

                    # Prediction of previously unsafe points based on that
                    mean2, var2 = gp.predict_noiseless(self.inputs[~self.S])

                    # Remove the fake data point from the GP again
                    self._remove_last_data_point(gp=gp)

                    mean2 = mean2.squeeze()
                    var2 = var2.squeeze()
                    l2 = mean2 - beta * np.sqrt(var2)

                    # If any unsafe lower bound is suddenly above fmin then
                    # the point is an expander
                    G_safe[index] = np.any(l2 >= self.fmin[i])

                    # Break if one safety GP is not an expander
                    if not G_safe[index]:
                        break

            # Since we sorted by uncertainty and only the most
            # uncertain element gets picked by SafeOpt anyways, we can
            # stop after we found the first one
            if G_safe[index] and not full_sets:
                break

        # Update safe set (if full_sets is False this is at most one point
        self.G[s] = G_safe

    def get_new_query_point(self, ucb=False):
        """
        Compute a new point at which to evaluate the function.

        Parameters
        ----------
        ucb: bool
            If True the safe-ucb criteria is used instead.

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        if not np.any(self.S):
            raise EnvironmentError('There are no safe points to evaluate.')

        if ucb:
            max_id = np.argmax(self.Q[self.S, 1])
            x = self.inputs[self.S, :][max_id, :]
        else:
            # Get lower and upper bounds
            l = self.Q[:, ::2]
            u = self.Q[:, 1::2]

            MG = np.logical_or(self.M, self.G)
            value = np.max((u[MG] - l[MG]) / self.scaling, axis=1)
            x = self.inputs[MG, :][np.argmax(value), :]

        if self.num_contexts:
            return x[:-self.num_contexts]
        else:
            return x

    def optimize(self, context=None, ucb=False):
        """Run Safe Bayesian optimization and get the next parameters.

        Parameters
        ----------
        context: ndarray
            A vector containing the current context
        ucb: bool
            If True the safe-ucb criteria is used instead.

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        # Update confidence intervals based on current estimate
        self.update_confidence_intervals(context=context)

        # Update the sets
        if ucb:
            self.compute_safe_set()
        else:
            self.compute_sets()

        return self.get_new_query_point(ucb=ucb)

    def get_maximum(self, context=None):
        """
        Return the current estimate for the maximum.

        Parameters
        ----------
        context: ndarray
            A vector containing the current context

        Returns
        -------
        x - ndarray
            Location of the maximum
        y - 0darray
            Maximum value

        Notes
        -----
        Uses the current context and confidence intervals!
        Run update_confidence_intervals first if you recently added a new data
        point.
        """
        self.update_confidence_intervals(context=context)

        # Compute the safe set (that's cheap anyways)
        self.compute_safe_set()

        # Return nothing if there are no safe points
        if not np.any(self.S):
            return None

        l = self.Q[self.S, 0]

        max_id = np.argmax(l)
        return (self.inputs[self.S, :][max_id, :-self.num_contexts or None],
                l[max_id])


class SafeOptSwarm(GaussianProcessOptimization):
    """SafeOpt for larger dimensions using a Swarm Optimization heuristic.

    Note that it doesn't support the use of a Lipschitz constant nor contextual
    optimization.

    You can set your logging level to INFO to get more insights on the
    optimization process.

    Parameters
    ----------
    gp: GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints.
    fmin: list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted)
    bounds: pair of floats or list of pairs of floats
        If a list is given, then each pair represents the lower/upper bound in
        each dimension. Otherwise, we assume the same bounds for all
        dimensions. This is mostly important for plotting or to restrict
        particles to a certain domain.
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
        If a list, this represents the stopping criterion for all the gps.
        This ignores the scaling factor.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of
        each kernel. You should probably set this to "auto" unless your kernel
        is non-stationary
    swarm_size: int
        The number of particles in each of the optimization swarms

    Examples
    --------
    >>> from safeopt import SafeOptSwarm
    >>> import GPy
    >>> import numpy as np

    Define a Gaussian process prior over the performance

    >>> x = np.array([[0.]])
    >>> y = np.array([[1.]])
    >>> gp = GPy.models.GPRegression(x, y, noise_var=0.01**2)

    Initialize the Bayesian optimization and get new parameters to evaluate

    >>> opt = SafeOptSwarm(gp, fmin=[0.], bounds=[[-1., 1.]])
    >>> next_parameters = opt.optimize()

    Add a new data point with the parameters and the performance to the GP. The
    performance has normally be determined through an external function call.

    >>> performance = np.array([[1.]])
    >>> opt.add_new_data_point(next_parameters, performance)

    """

    def __init__(self, gp, fmin, bounds, beta=2, scaling='auto', threshold=0,
                 swarm_size=20,define_swarms=True):
        """Initialization, see `SafeOptSwarm`."""
        super(SafeOptSwarm, self).__init__(gp,
                                           fmin=fmin,
                                           beta=beta,
                                           num_contexts=0,
                                           threshold=threshold,
                                           scaling=scaling)

        # Safe set -> contains positions of the parameters
        self.S = np.asarray(self.gps[0].X)

        self.swarm_size = swarm_size
        self.max_iters = 100  # number of swarm iterations

        if not isinstance(bounds, list):
            self.bounds = [bounds] * self.S.shape[1]
        else:
            self.bounds = bounds

        # These are estimates of the best lower bound, and its location
        self.best_lower_bound = -np.inf
        # First point in the initial safe set is our greed point
        self.greedy_point = self.S[0, :]

        # Finds velocities for swarm optimization for each parameter and GP
        self.optimal_velocities = self.optimize_particle_velocity()

        swarm_types = ['greedy', 'maximizers', 'expanders']

        if define_swarms:
            # Define the 3 swarms, greedy, maximizers and expanders
            self.swarms = {swarm_type:
                           SwarmOptimization(
                               swarm_size,
                               self.optimal_velocities,
                               partial(self._compute_particle_fitness,
                                       swarm_type),
                               bounds=self.bounds)
                           for swarm_type in swarm_types}

    def optimize_particle_velocity(self):
        """Optimize the velocities of the particles.

        Note that this only works well for stationary kernels and constant mean
        functions. Otherwise the velocity depends on the position!

        Returns
        -------
        velocities: ndarray
            The estimated optimal velocities in each direction.
        """

        parameters = np.zeros((1, self.gp.input_dim), dtype=np.float)
        velocities = np.empty((len(self.gps), self.gp.input_dim),
                              dtype=np.float)

        for i, gp in enumerate(self.gps):
            for j in range(self.gp.input_dim):
                tmp_velocities = np.zeros((1, self.gp.input_dim),
                                          dtype=np.float)

                # lower and upper bounds on velocities
                upper_velocity = 1000.
                lower_velocity = 0.

                # Binary search over optimal velocities
                while True:
                    mid = (upper_velocity + lower_velocity) / 2
                    tmp_velocities[0, j] = mid
                    # Calculates k(parameters,tmp_velocities); -> As stationary kernel k(x,x+d) = k(0,d)
                    kernel_matrix = gp.kern.K(parameters, tmp_velocities)
                    covariance = kernel_matrix.squeeze() / self.scaling[i] ** 2

                    # Make sure the correlation is in the sweet spot
                    velocity_enough = covariance > 0.94
                    not_too_fast = covariance < 0.95

                    if not_too_fast:
                        upper_velocity = mid
                    elif velocity_enough:
                        lower_velocity = mid

                    if ((not_too_fast and velocity_enough) or
                            upper_velocity - lower_velocity < 1e-5):
                        break

                # Store optimal velocity
                velocities[i, j] = mid

        # Select the minimal velocity (for the toughest safety constraint)
        velocities = np.min(velocities, axis=0)

        # Scale for number of parameters (this might not be so clever if they
        # are all independent, additive kernels).
        velocities /= np.sqrt(self.gp.input_dim)
        return velocities

    def _compute_penalty(self, slack):
        """Return the penalty associated to a constraint violation.

        The penalty is a piecewise linear function that is nonzero only if the
        safety constraints are violated. This penalty encourages particles to
        stay within the safe set.

        Parameters
        ----------
        slack: ndarray
            A vector corresponding to how much the constraint was violated.

        Returns
        -------
        penalties - ndarray
            The value of the penalties
        """
        penalties = np.atleast_1d(np.clip(slack, None, 0))

        penalties[(slack < 0) & (slack > -0.001)] *= 2
        penalties[(slack <= -0.001) & (slack > -0.1)] *= 5
        penalties[(slack <= -0.1) & (slack > -1)] *= 10

        slack_id = slack < -1
        penalties[slack_id] = -300 * penalties[slack_id] ** 2
        return penalties

    def _compute_particle_fitness(self, swarm_type, particles):
        """
        Return the value of the particles and the safety information.

        Parameters
        ----------
        particles : ndarray
            A vector containing the coordinates of the particles
        swarm_type : string
            A string corresponding to the swarm type. It can be any of the
            following strings:

                * 'greedy' : Optimal value(best lower bound).
                * 'expander' : Expanders (lower bound close to constraint)
                * 'maximizer' : Maximizers (Upper bound better than best l)
                * 'safe_set' : Only check the safety of the particles
        Returns
        -------
        values : ndarray
            The values of the particles
        global_safe : ndarray
            A boolean mask indicating safety status of all particles
            (note that in the case of a greedy swarm, this is not computed and
            we return a True mask)
        """
        beta = self.beta(self.t)

        # classify the particle's function values
        mean, var = self.gps[0].predict_noiseless(particles)
        mean = mean.squeeze()
        std_dev = np.sqrt(var.squeeze())

        # compute the confidence interval
        lower_bound = np.atleast_1d(mean - beta * std_dev)
        upper_bound = np.atleast_1d(mean + beta * std_dev)

        # the greedy swarm optimizes for the lower bound
        if swarm_type == 'greedy':
            return lower_bound, np.broadcast_to(True, len(lower_bound))

        # value we are optimizing for. Expanders and maximizers seek high
        # variance points
        values = std_dev / self.scaling[0]

        #
        is_safe = swarm_type == 'safe_set'
        is_expander = swarm_type == 'expanders'
        is_maximizer = swarm_type == 'maximizers'

        if is_safe:
            interest_function = None
        else:
            if is_expander:
                # For expanders, the interest function is updated depending on
                # the lower bounds
                interest_function = (len(self.gps) *
                                     np.ones(np.shape(values), dtype=np.float))
            elif is_maximizer:
                improvement = upper_bound - self.best_lower_bound
                interest_function = expit(10 * improvement / self.scaling[0])
            else:
                # unknown particle type (shouldn't happen)
                raise AssertionError("Invalid swarm type")

        # boolean mask that tell if the particles are safe according to all gps
        global_safe = np.ones(particles.shape[0], dtype=np.bool)
        total_penalty = np.zeros(particles.shape[0], dtype=np.float)

        for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
            # Only recompute confidence intervals for constraints
            if i > 0:
                # classify using the current GP
                mean, var = gp.predict_noiseless(particles)
                mean = mean.squeeze()
                std_dev = np.sqrt(var.squeeze())
                lower_bound = mean - beta * std_dev

                values = np.maximum(values, std_dev / scaling)

            # if the current GP has no safety constrain, we skip it
            if self.fmin[i] == -np.inf:
                continue

            slack = np.atleast_1d(lower_bound - self.fmin[i])

            # computing penalties
            global_safe &= slack >= 0

            # Skip cost update for safety evaluation
            if is_safe:
                continue

            # Normalize the slack somewhat
            slack /= scaling

            total_penalty += self._compute_penalty(slack)

            if is_expander:
                # check if the particles are expanders for the current gp
                interest_function *= norm.pdf(slack, scale=0.2)

        # this swarm type is only interested in knowing whether the particles
        # are safe.
        if is_safe:
            return lower_bound, global_safe

        # add penalty
        values += total_penalty

        # apply the mask for current interest function
        values *= interest_function

        return values, global_safe

    def get_new_query_point(self, swarm_type):
        """
        Compute a new point at which to evaluate the function.

        This function relies on a Particle Swarm Optimization (PSO) to find the
        optimum of the objective function (which depends on the swarm type).

        Parameters
        ----------
        swarm_type: string
            This parameter controls the type of point that should be found. It
            can take one of the following values:

                * 'expanders' : find a point that increases the safe set
                * 'maximizers' : find a point that maximizes the objective
                                 function within the safe set.
                * 'greedy' : retrieve an estimate of the best currently known
                             parameters (best lower bound).

        Returns
        -------
        global_best: np.array
            The next parameters that should be evaluated.
        max_std_dev: float
            The current standard deviation in the point to be evaluated.
        """
        beta = self.beta(self.t)
        safe_size, input_dim = self.S.shape

        # Make sure the safe set is still safe
        _, safe = self._compute_particle_fitness('safe_set', self.S)

        num_safe = safe.sum()
        if num_safe == 0:
            raise RuntimeError('The safe set is empty.')

        # Prune safe set if points in the discrete approximation of the safe
        # ended up being unsafe, but never prune below swarm size to avoid
        # empty safe set.
        if num_safe >= self.swarm_size and num_safe != len(safe):
            # Warn that the safe set has decreased
            logging.warning("Warning: {} unsafe points removed. "
                            "Model might be violated"
                            .format(np.count_nonzero(~safe)))

            # Remove unsafe points
            self.S = self.S[safe]
            safe_size = self.S.shape[0]

        # initialize particles
        if swarm_type == 'greedy':
            # we pick particles u.a.r in the safe set
            random_id = np.random.randint(safe_size, size=self.swarm_size - 3)
            best_sampled_point = np.argmax(self.gp.Y)

            # Particles are drawn at random from the safe set, but include the
            # - Previous greedy estimate
            # - last point
            # - best sampled point
            particles = np.vstack((self.S[random_id, :],
                                   self.greedy_point,
                                   self.gp.X[-1, :],
                                   self.gp.X[best_sampled_point]))
        else:
            # we pick particles u.a.r in the safe set
            random_id = np.random.randint(safe_size, size=self.swarm_size)
            particles = self.S[random_id, :]

        # Run the swarm optimization
        swarm = self.swarms[swarm_type]
        swarm.init_swarm(particles)
        swarm.run_swarm(self.max_iters)

        # expand safe set
        if swarm_type != 'greedy':
            num_added = 0

            # compute correlation between new candidates and current safe set
            covariance = self.gp.kern.K(swarm.best_positions,
                                        np.vstack((self.S,
                                                   swarm.best_positions)))
            covariance /= self.scaling[0] ** 2

            initial_safe = len(self.S)
            n, m = np.shape(covariance)

            # this mask keeps track of the points that we have added in the
            # safe set to account for them when adding a new point
            mask = np.zeros(m, dtype=np.bool)
            mask[:initial_safe] = True

            for j in range(n):
                # make sure correlation with old points is relatively low
                if np.all(covariance[j, mask] <= 0.95):
                    self.S = np.vstack((self.S, swarm.best_positions[[j], :]))
                    num_added += 1
                    mask[initial_safe + j] = True

            logging.debug("At the end of swarm {}, {} points were appended to"
                          " the safeset".format(swarm_type, num_added))
        else:
            # check whether we found a better estimate of the lower bound
            mean, var = self.gp.predict_noiseless(self.greedy_point[None, :])
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())

            lower_bound = mean - beta * std_dev
            if lower_bound < np.max(swarm.best_values):
                self.greedy_point = swarm.global_best.copy()

        if swarm_type == 'greedy':
            return swarm.global_best.copy(), np.max(swarm.best_values)

        # compute the variance of the point picked
        var = np.empty(len(self.gps), dtype=np.float)
        # max_std_dev = 0.
        for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
            var[i] = gp.predict_noiseless(swarm.global_best[None, :])[1]

        return swarm.global_best, np.sqrt(var)

    def optimize(self, ucb=False):
        """Run Safe Bayesian optimization and get the next parameters.

        Parameters
        ----------
        ucb: bool
            Whether to only compute maximizers (best upper bound).

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        # compute estimate of the lower bound
        self.greedy, self.best_lower_bound = self.get_new_query_point('greedy')

        # Run both swarms:
        x_maxi, std_maxi = self.get_new_query_point('maximizers')
        if ucb:
            logging.info('Using ucb criterion.')
            return x_maxi

        x_exp, std_exp = self.get_new_query_point('expanders')

        # Remove expanders below threshold or without safety constraint.
        std_exp[(std_exp < self.threshold) | (self.fmin == -np.inf)] = 0

        # Apply scaling
        std_exp /= self.scaling
        std_exp = np.max(std_exp)

        std_maxi = std_maxi[0] / self.scaling[0]

        logging.info("The best maximizer has std. dev. %f" % std_maxi)
        logging.info("The best expander has std. dev. %f" % std_exp)
        logging.info("The greedy estimate of lower bound has value %f" %
                     self.best_lower_bound)

        if std_maxi > std_exp:
            return x_maxi
        else:
            return x_exp

    def get_maximum(self):
        """
        Return the current estimate for the maximum.

        Returns
        -------
        x : ndarray
            Location of the maximum
        y : 0darray
            Maximum value

        """
        maxi = np.argmax(self.gp.Y)
        return self.gp.X[maxi, :], self.gp.Y[maxi]


class GoSafeSwarm_Contextual(SafeOptSwarm):

    """GoSafe for larger dimensions using a Swarm Optimization heuristic without exploration in state space.

    Note that it doesn't support the use of a Lipschitz constant nor contextual
    optimization.

    You can set your logging level to INFO to get more insights on the
    optimization process.

    Parameters
    ----------
    gp: GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints.
    gp_full: GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints. Unlike, gp, this contains also the states
        as input paramters. It is used to evaluate the boundary condition
    fmin: list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted)
    bounds: pair of floats or list of pairs of floats
        If a list is given, then each pair represents the lower/upper bound in
        each dimension. Otherwise, we assume the same bounds for all
        dimensions. This is mostly important for plotting or to restrict
        particles to a certain domain.
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
        If a list, this represents the stopping criterion for all the gps.
        This ignores the scaling factor.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of
        each kernel. You should probably set this to "auto" unless your kernel
        is non-stationary
    swarm_size: int
        The number of particles in each of the optimization swarms
    max_S1_steps: int
        Maximum number of consecutive S1 steps
    max_S3_steps: int
        Maximum number of consecutive S3 steps
    eta_L: double
        Lower tolerance used for the boundary conditon
    eta_u: double
        Upper tolerance used for the boundary conditon
    eps: double
        Maximum precision for convergence
    max_data_size: int
        Maximum number of data points of states other than x_0 that
        the full GP can contain
    reset_size: int
        Number of data points we choose for our subset (only considering states
        other than x0).
    boundary_thresshold_u: double
        Upper thresshold used for checking the boundary condition-> Not at boundary
        if current state x has x_s with k(x,x_s) > boundary_thresshold_u and l_n(x_s,a) >= eta_L
    boundary_thresshold_l: double
        Lower thresshold used for checking the boundary condition-> Not at boundary
        if current state x has x_s with k(x,x_s) >= boundary_thresshold_L and l_n(x_s,a) >= eta_U



    Examples
    --------
    >>> from safeopt import GoSafeSwarm_Contextual
    >>> import GPy
    >>> import numpy as np

    Define a Gaussian process prior over the performance
    >>> x_0=0.5
    >>> x = np.array([[0.,x_0]])
    >>> a=np.array([[0]])
    >>> y = np.array([[1.]])
    >>> gp = GPy.models.GPRegression(a, y, noise_var=0.01**2)
    >>> gp_full=GPy.models.GPRegression(x, y, noise_var=0.01**2)

    Initialize the Bayesian optimization and get new parameters to evaluate

    >>> opt = GoSafeSwarm_Contextual(gp,gp_full,fmin=[0.], bounds=[[-1., 1.]],x_0=np.asarray([[x_0]]))
    >>> next_parameters = opt.optimize()

    Add a new data point with the parameters and the performance to the GP. The
    performance has normally be determined through an external function call.

    >>> performance = np.array([[1.]])
    >>> params=np.hstack([next_parameters.reshape(1,-1),x_0]).reshape(1,-1)
    >>> opt.add_new_data_point(params, performance)

    """

    def __init__(self, gp, gp_full,fmin, bounds,x_0,L_states,beta=2, scaling='auto', threshold=0,
                 swarm_size=20,max_S1_steps=30,max_S3_steps=10,eta_L=0.1,eta_u=np.inf,eps=0.1,max_data_size=400,reset_size=200,boundary_thresshold_u=0.95,boundary_thresshold_l=0.88):

        assert len(gp_full) == len(gp), 'Full gp must have the same dimension as the parameter gp'
        self.gp_full=gp_full
        # Initialize all SafeOptSwarm params
        super(GoSafeSwarm_Contextual, self).__init__(gp=gp,
                                          fmin=fmin,
                                          bounds=bounds,
                                          beta=beta,
                                          scaling=scaling,
                                          threshold=threshold,
                                          swarm_size=swarm_size,
                                          define_swarms=True)


        #self.gp_full=gp_full

        # Read state information
        self.state_dim = np.shape(x_0)[0]
        self.state_idx = list(range(self.gp_full[0].input_dim - self.state_dim, self.gp_full[0].input_dim))
        self.x_0 = x_0.T


        self.action_dim=self.gp.input_dim
        assert self.action_dim+self.state_dim == self.gp_full[0].input_dim, 'state dim + action dim != Full GP input dim'
        # Define swarm for S3, global search. No velocity constraint required as safety is not asked
        S3_velocities = self.optimal_velocities.copy()
        S3_velocities = S3_velocities * 10
        self.swarms['S3']=SwarmOptimization(swarm_size,S3_velocities,partial(self._compute_particle_fitness,'S3'),bounds=bounds)
        self.criterion = 'init'
        # Boolean used to indicate if we should perturb initialized particles for further exploration before running swarm optimization
        self.perturb_particles = True
        # Define counters for x_0 in the subset of data and all data points recorded
        x_0_idx=np.asarray(range(len(self.S)))
        self.x_0_idx_gp_full = x_0_idx.copy()
        # All combinations of x_0 in all the recorded data
        self.x_0_idx_full_data = x_0_idx.copy()
        # Boolean to indicate  if a data point is in the full GP or not
        self.in_gp_full = np.ones(x_0_idx.shape[0], dtype=bool)

        # Define optimization parameters
        self.s1_steps = 0
        self.s3_steps = 0
        self.max_S1_steps = max_S1_steps
        self.max_S3_steps = max_S3_steps
        self.switch=False
        # Define tolerance parameter
        self.eta_L = eta_L
        self.eta_u = eta_u
        self.use_marginal_set = self.eta_u != np.inf
        # Stores information from failed experiments (failed initial state action pair and the state at which we hit the boundary)
        self.Failed_experiment_list = []
        self.Failed_state_list = []
        # Fast distance approximate used to reduce computational cost while evaluating the covariances between points
        # If true, it only takes the kernel of f into consideration instead of all the GPs
        self.fast_distance_approximation = False
        # Define set numbers to indicate globally explored regions
        self.set_number = np.zeros(self.S.shape[0], dtype=int)
        self.current_set_number = 0

        self.eps = eps
        self.data_size_max = max_data_size
        self.N_reset = reset_size
        # Lower bounds for all the data points collected so far
        self.lower_bound = np.ones([self._x.shape[0], len(self.gps)]) * -np.inf
        # Interior points+marginal: Points that fulfill the boundary condition
        self.interior_points=np.zeros(self.lower_bound.shape[0]) # Points with l_n >= eta_u
        self.marginal_points = np.zeros(self.lower_bound.shape[0])# Points with l_n >= eta_L
        # Update lower bounds and calculate interior and marginal points
        self.Update_data_lower_bounds()

        # Boundary thresshold, used to evaluate boundary condition
        self.boundary_thresshold_u = boundary_thresshold_u
        self.boundary_thresshold_l = boundary_thresshold_l
        # Indicates if safe action should be returned if the boundary is hit.
        self.return_safe_action=True
        # Lengthscales of the states
        self.L_states=L_states
        # Estimate maximum distance we can go away from a point such that covariance is >=boundary_thresshold_u
        self.state_velocities_u=self.optimal_state_velocity(self.boundary_thresshold_u)
        self.state_squaraed_dist_u=np.sum(np.square(self.state_velocities_u/self.L_states))


        if self.use_marginal_set:
            # Estimate maximum distance we can go away from a point such that covariance is >=boundary_thresshold_l
            self.state_velocities_l = self.optimal_state_velocity(self.boundary_thresshold_l)
            self.state_squaraed_dist_l = np.sum(np.square(self.state_velocities_l / self.L_states))
            # Beware: area of confusion -> state_squaraed_dist_u corresponds to distance associated with boundary_thresshold_u.
            # Hence by definition state_squaraed_dist_u<state_squaraed_dist_l.

        self.encourage_jumps = True
        self.jump_frequency=20
        self.fast_safe_action = True


    def _get_initial_xy(self):
        """Get the initial x/y data from the GPs."""
        self._x = self.gp_full[0].X
        y = [self.gp_full[0].Y]
        for gp in self.gp_full[1:]:
            if np.allclose(self._x, gp.X):
                y.append(gp.Y)
            else:
                raise NotImplemented('The GPs have different measurements.')

        self._y = np.concatenate(y, axis=1)

    
    def _seed(self, seed=None):
        '''
        Sets the numpy random seed
        :param seed:
        :return:
        '''
        if seed is not None:
            np.random.seed(seed)


    def optimal_state_velocity(self,boundary_thresshold):
        """Get the velocities of the states such that covariance is greater than boundary_thresshold.

        Note that this only works well for stationary kernels and constant mean
        functions. Otherwise the velocity depends on the position!

        Returns
        -------
        velocities: ndarray
            The estimated optimal velocities in each direction of the state.
        """

        parameters = np.zeros((1, self.gp_full[0].input_dim), dtype=np.float)
        velocities = np.empty((len(self.gp_full), self.state_dim),
                              dtype=np.float)

        for i, gp in enumerate(self.gp_full):
            for j in range(self.action_dim,self.gp_full[0].input_dim):
                tmp_velocities = np.zeros((1, self.gp_full[0].input_dim),
                                          dtype=np.float)

                # lower and upper bounds on velocities
                upper_velocity = 1000.
                lower_velocity = 0.

                # Binary search over optimal velocities
                while True:
                    mid = (upper_velocity + lower_velocity) / 2
                    tmp_velocities[0, j] = mid
                    # Calculates k(parameters,tmp_velocities); -> As stationary kernel k(x,x+d) = k(0,d)
                    kernel_matrix = gp.kern.K(parameters, tmp_velocities)
                    covariance = kernel_matrix.squeeze() / self.scaling[i] ** 2

                    # Make sure the correlation is in the sweet spot
                    velocity_enough = covariance > boundary_thresshold-0.01
                    not_too_fast = covariance < boundary_thresshold

                    if not_too_fast:
                        upper_velocity = mid
                    elif velocity_enough:
                        lower_velocity = mid

                    if ((not_too_fast and velocity_enough) or
                            upper_velocity - lower_velocity < 1e-5):
                        break

                # Store optimal velocity
                velocities[i, j-self.action_dim] = mid


        if self.fmin[0]==-np.inf:
            # Select the minimal velocity (for the toughest safety constraint)
            velocities = np.min(velocities[1:,:], axis=0)
        else:
            velocities = np.min(velocities, axis=0)


        # Scale for number of parameters (this might not be so clever if they
        # are all independent, additive kernels).
        velocities /= np.sqrt(self.gp_full[0].input_dim)
        return velocities


    def compute_particle_distance(self, particles, X_data, full=False):
        """
        Return the covariance between particles and X_data

        Parameters
        ----------

        particles : ndarray
            A vector containing parameter and states for which we
            want to calculate the covariance

        X_data  : ndarray
            A vector containing parameter and states for which we
            want to calculate the covariance

        full    : bool
            If true we return a pointwise minimum
            of the covariances with respect to all GPs
            else we return the highest covariance of
            X_data for each particle


        Returns
        -------
        covariance_mat: The full covariance matrix using all the GPs if full is True
        covariance: The maximum covariance between the particles and the data if full is false

        """

        # If full, we want to return the whole matrix
        if full:
            # fast_distance_approximation just considers the GP of the objective
            # If false, we take all GPs into account
            if not self.fast_distance_approximation:

                covariance_mat = np.ones([particles.shape[0], X_data.shape[0]]) * np.inf

                for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                    # Do not consider GP if it has no constraint
                    if self.fmin[i] == -np.inf:
                        continue
                    # Take the pointwise minimum over each GP
                    covariance_mat = np.minimum(covariance_mat, gp.kern.K(particles, X_data) / scaling ** 2)

                return covariance_mat

            else:
                # Just return the covariance matrix of the objective
                covariance_mat = self.gp.kern.K(particles, X_data)
                covariance_mat /= self.scaling[0] ** 2
                return covariance_mat

        # Here we want to find the highest covariance between the particles
        # and points in X_data
        else:
            # fast_distance_approximation just considers the GP of the objective
            # If false, we take all GPs into account
            if not self.fast_distance_approximation:
                covariance = np.zeros(particles.shape[0])
                for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                    # Neglect point if no constraint
                    if self.fmin[i] == -np.inf:
                        continue
                    # Determine covariance
                    covariance_mat = gp.kern.K(particles, X_data)
                    covariance_mat /= scaling ** 2
                    # Find maximum covariance for each particle
                    covariance_mat = np.max(covariance_mat, axis=1)

                    # Compare to the covariances we have so far and
                    # take the maximum
                    covariance = np.maximum(covariance, covariance_mat)

            else:
                # Determine covariance using solely the objective
                covariance = self.gp.kern.K(particles, X_data)
                covariance /= self.scaling[0] ** 2
                covariance = np.max(covariance, axis=1)
            return covariance

    def _compute_particle_fitness(self, swarm_type, particles):
        """
        Return the value of the particles and the safety information.
        Same as for SafeOptSwarm but takes min of the lower bounds for the interest function.

        Parameters
        ----------
        particles : ndarray
            A vector containing the coordinates of the particles
        swarm_type : string
            A string corresponding to the swarm type. It can be any of the
            following strings:

                * 'greedy' : Optimal value(best lower bound).
                * 'expander' : Expanders (lower bound close to constraint)
                * 'maximizer' : Maximizers (Upper bound better than best l)
                * 'safe_set' : Only check the safety of the particles
        Returns
        -------
        values : ndarray
            The values of the particles
        global_safe : ndarray
            A boolean mask indicating safety status of all particles
            (note that in the case of a greedy swarm, this is not computed and
            we return a True mask)
        """
        beta = self.beta(self.t)

        # classify the particle's function values
        mean, var = self.gps[0].predict_noiseless(particles)
        mean = mean.squeeze()
        std_dev = np.sqrt(var.squeeze())

        # compute the confidence interval
        lower_bound = np.atleast_1d(mean - beta * std_dev)
        upper_bound = np.atleast_1d(mean + beta * std_dev)

        # the greedy swarm optimizes for the lower bound
        if swarm_type == 'greedy':
            return lower_bound, np.broadcast_to(True, len(lower_bound))


        # value we are optimizing for. Expanders and maximizers seek high
        # variance points
        values = std_dev / self.scaling[0]

        #
        is_safe = swarm_type == 'safe_set'
        is_expander = swarm_type == 'expanders'
        is_maximizer = swarm_type == 'maximizers'
        is_S3 = swarm_type == 'S3'
        if is_safe:
            interest_function = None
        else:
            if is_expander:
                # For expanders, the interest function is updated depending on
                # the lower bounds
                interest_function = (len(self.gps) *
                                     np.ones(np.shape(values), dtype=np.float))
                slack_min = np.ones(len(particles)) * np.inf
            elif is_maximizer:
                improvement = upper_bound - self.best_lower_bound
                interest_function = expit(10 * improvement / self.scaling[0])
            elif is_S3:
                # Take the maximum covariance (minimum distance) between particles and the failed set + data points + safe set
                interest_function = (len(self.gps) *
                                     np.ones(np.shape(values), dtype=np.float))
                X_data = np.vstack((self.gp.X, self.S))
                covariance = self.compute_particle_distance(particles, X_data)
                if self.Failed_experiment_list:

                    failed_experiments = np.asarray(self.Failed_experiment_list.copy()).reshape(-1,self.action_dim)
                    #print('Failed Experiments',failed_experiments)
                    covariance_failedset = self.compute_particle_distance(particles,failed_experiments)
                    covariance = np.maximum(covariance, covariance_failedset)
                #print("In here")
                interest_function *= np.exp(-5 * covariance)


            else:
                # unknown particle type (shouldn't happen)
                raise AssertionError("Invalid swarm type")

        # boolean mask that tell if the particles are safe according to all gps
        global_safe = np.ones(particles.shape[0], dtype=np.bool)
        total_penalty = np.zeros(particles.shape[0], dtype=np.float)

        for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
            # Only recompute confidence intervals for constraints
            if i > 0:
                # classify using the current GP
                mean, var = gp.predict_noiseless(particles)
                mean = mean.squeeze()
                std_dev = np.sqrt(var.squeeze())
                lower_bound = mean - beta * std_dev

                values = np.maximum(values, std_dev / scaling)

            # if the current GP has no safety constrain, we skip it
            if self.fmin[i] == -np.inf:
                continue

            if is_S3:
                continue
            slack = np.atleast_1d(lower_bound - self.fmin[i])

            # computing penalties
            global_safe &= slack >= 0

            # Skip cost update for safety evaluation
            if is_safe:
                continue

            # Normalize the slack somewhat
            slack /= scaling

            total_penalty += self._compute_penalty(slack)

            if is_expander:
                # check if the particles are expanders for the current gp
                #interest_function *= norm.pdf(slack, scale=0.2)
                slack_min = np.minimum(slack_min, slack)

        if is_expander:
            interest_function *= norm.pdf(slack_min, scale=0.2)

        # this swarm type is only interested in knowing whether the particles
        # are safe.
        if is_safe:
            return lower_bound, global_safe

        # add penalty
        values += total_penalty

        # apply the mask for current interest function
        values *= interest_function

        return values, global_safe

    def get_new_query_point(self, swarm_type):
        """
        Compute a new point at which to evaluate the function.

        This function relies on a Particle Swarm Optimization (PSO) to find the
        optimum of the objective function (which depends on the swarm type).
        This is essentially the same as for SafeOptSwarm, it only includes
        the S3 swarm.

        Parameters
        ----------
        swarm_type: string
            This parameter controls the type of point that should be found. It
            can take one of the following values:

                * 'expanders' : find a point that increases the safe set
                * 'maximizers' : find a point that maximizes the objective
                                 function within the safe set.
                * 'greedy' : retrieve an estimate of the best currently known
                             parameters (best lower bound).
                * 'S3': Optimization global

        Returns
        -------
        global_best: np.array
            The next parameters that should be evaluated.
        max_std_dev: float
            The current standard deviation in the point to be evaluated.
        """
        beta = self.beta(self.t)
        safe_size, input_dim = self.S.shape

        # Make sure the safe set is still safe
        _, safe = self._compute_particle_fitness('safe_set', self.S)

        num_safe = safe.sum()
        if num_safe == 0:
            raise RuntimeError('The safe set is empty.')

        # Prune safe set if points in the discrete approximation of the safe
        # ended up being unsafe, but never prune below swarm size to avoid
        # empty safe set.
        if num_safe >= self.swarm_size and num_safe != len(safe):
            # Warn that the safe set has decreased
            logging.warning("Warning: {} unsafe points removed. "
                            "Model might be violated"
                            .format(np.count_nonzero(~safe)))

            # Remove unsafe points
            self.S = self.S[safe]
            self.set_number = self.set_number[safe]
            safe_size = self.S.shape[0]

        # initialize particles
        if swarm_type == 'greedy':
            # we pick particles u.a.r in the safe set
            random_id = np.random.randint(safe_size, size=self.swarm_size - 3)
            best_sampled_point = np.argmax(self.gp.Y)

            # Particles are drawn at random from the safe set, but include the
            # - Previous greedy estimate
            # - last point
            # - best sampled point
            particles = np.vstack((self.S[random_id, :],
                                   self.greedy_point,
                                   self.gp.X[-1, :],
                                   self.gp.X[best_sampled_point]))
        else:
            if swarm_type=='S3':
                # sample points at random from the domain
                bound = np.array(self.bounds)
                action = np.random.uniform(low=bound[:, 0], high=bound[:, 1],
                                           size=[self.swarm_size, self.action_dim])
                particles = action.reshape(-1, self.action_dim)
                current_set = self.current_set_number
            else:
                # If new region was discovered, sample forcefully from the new area
                indexes = np.where(self.set_number == self.current_set_number)[0]
                if len(indexes)>0:
                    # If yes, randomly sample from this set
                    random_id = np.random.choice(indexes, size=self.swarm_size)
                    current_set = self.current_set_number
                    particles = self.S[random_id, :]
                else:
                    # IF we do not have any points from the latest safe set
                    # Sample points from the most recent safe set available for
                    # x_0
                    current_set = np.max(self.set_number)
                    indexes = np.where(self.set_number == current_set)[0]
                    random_id = np.random.choice(indexes, size=self.swarm_size)
                    particles = self.S[random_id, :]
                if self.perturb_particles:
                    u = np.random.uniform(-1, 1, size=particles.shape)
                    particles = particles + u * self.optimal_velocities
                    bound = np.array(self.bounds)
                    particles = np.clip(particles, bound[:,0], bound[:,1])


        # Run the swarm optimization
        swarm = self.swarms[swarm_type]
        swarm.init_swarm(particles)
        swarm.run_swarm(self.max_iters)

        # Add new points to our safe set adaptively
        if  swarm_type != 'greedy' and swarm_type != 'S3':
            num_added = 0

            # compute correlation between new candidates and current safe set
            covariance = self.gp.kern.K(swarm.best_positions,
                                        np.vstack((self.S,
                                                   swarm.best_positions)))
            covariance /= self.scaling[0] ** 2

            initial_safe = len(self.S)
            n, m = np.shape(covariance)

            # this mask keeps track of the points that we have added in the
            # safe set to account for them when adding a new point
            mask = np.zeros(m, dtype=np.bool)
            mask[:initial_safe] = True

            for j in range(n):
                # make sure correlation with old points is relatively low
                if np.all(covariance[j, mask] <= 0.95):
                    self.S = np.vstack((self.S, swarm.best_positions[[j], :]))
                    # Monitor set number for each point in the safe set
                    self.set_number = np.vstack((self.set_number, current_set))
                    num_added += 1
                    mask[initial_safe + j] = True

            logging.debug("At the end of swarm {}, {} points were appended to"
                          " the safeset".format(swarm_type, num_added))

        elif swarm_type=='greedy':
            # check whether we found a better estimate of the lower bound
            mean, var = self.gp.predict_noiseless(self.greedy_point[None, :])
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())

            lower_bound = mean - beta * std_dev
            if lower_bound < np.max(swarm.best_values):
                self.greedy_point = swarm.global_best.copy()
            return swarm.global_best.copy(), np.max(swarm.best_values)

        # compute the variance of the point picked
        var = np.empty(len(self.gps), dtype=np.float)
        for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
            var[i] = gp.predict_noiseless(swarm.global_best[None, :])[1]

        return swarm.global_best, np.sqrt(var)




    def optimize(self, ucb=False):
        """Run Safe Bayesian optimization and get the next parameters.

        Parameters
        ----------
        ucb: bool
            Whether to only compute maximizers (best upper bound).

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        # Update lower bounds of all particles
        self.Update_data_lower_bounds()
        # Boolean to check if we have any backup policy available
        no_back_up_policy=np.sum(self.interior_points)==0
        # compute estimate of the lower bound
        self.greedy, self.best_lower_bound = self.get_new_query_point('greedy')

        # If data size is too large, sample a subset
        if self.gp_full[0].X.shape[0]-self.gp_full[0].X[self.x_0_idx_gp_full,:].shape[0]>=self.data_size_max:
            self.select_gp_subset()

        # If encourage jump is true, after trying global search unsuccessfully reduce S1 steps
        max_S1_step=self.max_S1_steps
        if self.encourage_jumps:
            # S3 has not be successful, reduce S1 steps.
            max_S1_step=np.minimum(self.jump_frequency,self.max_S1_steps)
            if len(np.where(self.set_number == self.current_set_number)[0])>0:
                max_S1_step=self.max_S1_steps
        # Check if we have exceeded maximum number of steps for S1, if yes go to S3
        if self.s1_steps<max_S1_step:
            # Same as SafeOptSwarm
            self.criterion="S1"
            self.s1_steps+=1
            # Run both swarms:
            x_maxi, std_maxi = self.get_new_query_point('maximizers')
            if ucb:
                logging.info('Using ucb criterion.')
                return x_maxi

            x_exp, std_exp = self.get_new_query_point('expanders')

            # Remove expanders below threshold or without safety constraint.
            std_exp[(std_exp < self.threshold) | (self.fmin == -np.inf)] = 0

            # Apply scaling
            std_exp /= self.scaling
            std_exp = np.max(std_exp)

            std_maxi = std_maxi / self.scaling
            std_maxi = np.max(std_maxi)
            # Check if we have greater than eps uncertainty, if not go to S2
            if max(std_exp,std_maxi)>self.eps or no_back_up_policy:

                logging.info("The best maximizer has std. dev. %f" % std_maxi)
                logging.info("The best expander has std. dev. %f" % std_exp)
                logging.info("The greedy estimate of lower bound has value %f" %
                             self.best_lower_bound)

                if std_maxi >= std_exp:
                    return x_maxi.squeeze()
                else:
                    return x_exp.squeeze()

        # Run S3
        self.criterion = "S3"
        # If self.s3_steps=0 -> We are running S3 for the first time after exploring the previous set,
        # Update set number
        if self.s3_steps==0:
            self.current_set_number+=1

        self.s3_steps += 1
        # Check if we have exceeded maximum number of S3 steps, if yes reset all step counters, if no sample from S3 swarm
        self.switch= self.s3_steps>=self.max_S3_steps
        if self.switch:
            # Update counters
            self.s1_steps=0
            self.s3_steps=0
            self.switch = False


        # Find parameters for global search.
        x_exp, std_exp = self.get_new_query_point('S3')
        std_exp /= self.scaling
        std_exp = np.max(std_exp)
        logging.info("The best S3 point has std. dev. %f" % std_exp)
        return x_exp.squeeze()



    def add_new_data_point(self,x,y):
        '''
        Adds new data point to full gp and local gp (only for parameter). Slow as it adds data point 1 by 1.
        :param x: ndarray parameters + states
        :param y: ndarray objective and constraint values
        :return:

        '''
        is_x0=sum(x[0][self.state_idx] == self.x_0.squeeze()) == self.state_dim
        #beta=self.beta(self.t)
        a = x[:, :self.action_dim]
        # If 's3', add data point to safe set
        if self.criterion=="S3" and is_x0:
            self.criterion == "init"
            self.S = np.vstack((self.S, a))
            # Reset counters to run S1 for the newly discovered region.
            self.s3_steps=0
            self.s1_steps=0
            self.set_number = np.vstack((self.set_number, self.current_set_number))

        # Read out the parameters a
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        a=np.atleast_2d(a)
        # Update global data stores
        self._x = np.concatenate((self._x, x), axis=0)
        self._y = np.concatenate((self._y, y), axis=0)
        self.in_gp_full = np.append(self.in_gp_full, True)
        # If state corresponds to x0 add it to the parameter GP and full GP
        if is_x0:
            for i, gp in enumerate(self.gps):
                not_nan = ~np.isnan(y[:, i])
                if np.any(not_nan):
                    # Add data to GP and full GP
                    self._add_data_point(gp, a[not_nan, :], y[not_nan, [i]])
                    self._add_data_point(self.gp_full[i],x[not_nan, :], y[not_nan, [i]])

                #if self.fmin[i]==-np.inf:
                #    continue

            # Update initial condition place holders for full gp and full data.
            self.x_0_idx_gp_full = np.append(self.x_0_idx_gp_full, [self.gp_full[0].X.shape[0] - 1])
            self.x_0_idx_full_data = np.append(self.x_0_idx_full_data, [self._x.shape[0] - 1])

        else:
            # If not x_0, add data point to full gp
            for i,gp in enumerate(self.gp_full):
                not_nan = ~np.isnan(y[:, i])
                if np.any(not_nan):
                    self._add_data_point(gp, x[not_nan, :], y[not_nan, [i]])



    def add_data(self,x,y):
        '''
                Adds new data points to full gp and local gp (only for parameter).-> Fast
                :param x: ndarray parameters + states
                :param y: ndarray objective and constraint values
                :return:

        '''
        initial_size_full_data = self._x.shape[0]
        initial_size_gp_full = self.gp_full[0].X.shape[0]
        # Find where we observe x_0
        is_x0=np.sum(x[:,self.state_idx]==self.x_0.squeeze(),axis=1)==self.state_dim
        a = x[:, :self.action_dim]
        # If we ran global search successfully and first point was x0 (should be by default). Add newly discovered
        # safe parameter to the safe set
        if self.criterion=="S3" and is_x0[0]:
            self.criterion == "init"
            self.S = np.vstack((self.S, a[0,:]))
            # reset counters to explore with S1 the new region
            self.s3_steps=0
            self.s1_steps=0
            self.set_number = np.vstack((self.set_number, self.current_set_number))

        a=np.atleast_2d(a)
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        a=a[is_x0,:]
        y_x0=y[is_x0,:]

        # Update global data stores
        self._x = np.concatenate((self._x, x), axis=0)
        self._y = np.concatenate((self._y, y), axis=0)
        add_to_data = np.ones(x.shape[0], dtype=bool)
        self.in_gp_full = np.append(self.in_gp_full, add_to_data)
        # Loop over GPs
        for i, gp in enumerate(self.gps):
            not_nan = ~np.isnan(y[:, i])
            if np.any(not_nan):
                self.gp_full[i].set_XY(np.vstack([self.gp_full[i].X, x[not_nan, :]]),
                          np.vstack([self.gp_full[i].Y, y[not_nan, i].reshape(-1, 1)]))

                # Add data points corresponding to x_0 to the local GPs
                non_nan_x0=not_nan[is_x0]
                gp.set_XY(np.vstack([gp.X, a[non_nan_x0, :]]),
                          np.vstack([gp.Y, y_x0[non_nan_x0, i].reshape(-1, 1)]))

        new_points_idx_full_data = np.arange(start=initial_size_full_data,stop=self._x.shape[0])
        new_points_idx_gp_full = np.arange(start=initial_size_gp_full,stop=self.gp_full[0].X.shape[0])

        # Update initial condition place holders for full gp and full data.
        self.x_0_idx_gp_full = np.append(self.x_0_idx_gp_full, new_points_idx_gp_full[is_x0])
        self.x_0_idx_full_data = np.append(self.x_0_idx_full_data,new_points_idx_full_data[is_x0])

    def Update_data_lower_bounds(self):
        '''
        Updates the lower bounds for all the points in the dataset
        '''
        # Obtain beta and check how many new points have been collected
        beta=self.beta(self.t)
        difference=self._x.shape[0]-self.lower_bound.shape[0]
        self.lower_bound=np.vstack((self.lower_bound,np.ones([difference,len(self.gps)])*-np.inf))
        # Evaluate lower bound and update constrained set lower bounds if increasing.
        for i, gp in enumerate(self.gp_full):
            #if self.fmin[i]==-np.inf:
            #    continue
            mean, var = gp.predict_noiseless(self._x)
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())
            lower_bound = mean - beta * std_dev
            # Contained set based lower bounds
            self.lower_bound[:,i]=np.maximum(self.lower_bound[:, i], lower_bound)

        # Determine interior points: Points at which we will not hit the boundary
        constraint_idx=np.where(self.fmin != -np.inf)
        constraint_idx=np.asarray(constraint_idx).reshape(-1,1)
        slack=self.lower_bound[:,constraint_idx]-self.fmin[constraint_idx]
        # Determine interior sets
        self.interior_points=np.sum(slack-self.eta_L*self.scaling[constraint_idx]>0,axis=1)==constraint_idx.shape[0]
        self.interior_points = self.interior_points.squeeze()
        if self.use_marginal_set:
            self.marginal_points=np.sum(slack-self.eta_u*self.scaling[constraint_idx]<0,axis=1)==constraint_idx.shape[0]
            self.marginal_points = self.marginal_points.squeeze()
            self.marginal_points=np.logical_and(self.interior_points,self.marginal_points)


    def check_rollout(self, state, action):
        '''
        Checks rollout for the current state of the system
        :param state: ndarray, current state of the system
        :param action: ndarray, current action being applied
        :return:
        at_boundary: bool, True if we are the boundary,
        Fail: bool, Fail if experiment has failed
        a: ndarray, alternative safe action if we are at the boundary.
        '''
        at_boundary=False
        Fail = False
        if self.criterion == "S3":
            # check if we are at the boundary, if yes return an alternative safe action
            at_boundary, a = self.at_boundary(state)

        if not at_boundary:
            a=action

        return at_boundary,Fail,a

    def at_boundary(self,state):
        '''
        Checks if state is at the boundary of the safe set
        :param state: ndarray, indicating the current state of the system
        :return:
        at_boundary: bool, True if we are the boundary,
        a: ndarray, alternative safe action if we are at the boundary.
        '''
        # Read out all the interior states for which we would not hit the boundary
        interior_states = self._x[self.interior_points, :]
        # Check covariance between interior states and current state
        diff = interior_states[:, self.state_idx] - state
        diff = diff / self.L_states
        squared_dist = np.sum(np.square(diff), axis=1)
        if self.use_marginal_set:
            # For marginally safe points, we need smaller distances
            marginal_idx=self.marginal_points[self.interior_points]
            safe = squared_dist <= self.state_squaraed_dist_l
            # For marginally safe points, we need x to lie closer.
            safe[marginal_idx]&=squared_dist[marginal_idx]<=self.state_squaraed_dist_u
        else:
            safe=squared_dist<=self.state_squaraed_dist_u

        a_safe=None

        # If we even have 1 point that fulfills the boundary condition, we are not at the boundary.
        if np.sum(safe) > 0:
            at_boundary=False
        else:
            at_boundary=True
        # If at the boundary and an alternative safe action is asked, return the best action seen so far
        if at_boundary:
            # Add point to failed state list
            self.Failed_state_list.append(state)
            if self.return_safe_action:
                # Run swarm to return safe action, can be costly
                if not self.fast_safe_action:
                    idx_action = np.where(squared_dist == squared_dist.min())[0]
                    a_init=interior_states[idx_action,:self.action_dim].copy()
                    a_safe = self.find_constraint_max(a_init, state)
                else:
                    # Easier and works well in practice, find the closest state in the interior set, and return one of its safe actions.
                    idx_action=np.where(squared_dist==squared_dist.min())[0]
                    if len(idx_action)>1:
                        lb=self.lower_bound[self.interior_points,:]
                        idx_max=np.argmax(lb[idx_action,0])
                        idx_action=idx_action[idx_max]
                    #a_safe,f = self.get_maximum()
                    a_safe=interior_states[idx_action,:self.action_dim]
                #a_safe, f = self.get_maximum()

        return at_boundary,a_safe


    def add_boundary_points(self,x):
        # Add a point to failed set if it lead to failure/intervention was needed.
        self.Failed_experiment_list.append(x.copy())

    def update_boundary_points(self):
        """""
           Checks if failed experiments (epxeriments where we hit the boundary) 
           would still fail after updating GPs and safe sets.
           If yes, these experiments are removed from the GP. 
           
        """""

        # Does the same as check_boundary condition. But individually for each point in the failed set.

        # Check if we have any failed experiments
        if self.Failed_experiment_list:
            if self._x.shape[0]!=self.interior_points.shape[0]:
                self.Update_data_lower_bounds()
            # Loop over all the states in the failed_state_list and evaluate the boundary condition
            if self.use_marginal_set:
                for i,state in reversed(list(enumerate(self.Failed_state_list))):
                    interior_states = self._x[self.interior_points, :]
                    diff = interior_states[:, self.state_idx] - state
                    diff = diff / self.L_states
                    squared_dist = np.sum(np.square(diff), axis=1)
                    safe = squared_dist <= self.state_squaraed_dist_l
                    marginal_idx = self.marginal_points[self.interior_points]
                    safe[marginal_idx]&=squared_dist[marginal_idx]<=self.state_squaraed_dist_u
                    if np.sum(safe) > 0:
                        at_boundary = False
                    else:
                        at_boundary = True

            else:
                for i,state in reversed(list(enumerate(self.Failed_state_list))):
                    interior_states = self._x[self.interior_points, :]
                    diff = interior_states[:, self.state_idx] - state
                    diff = diff / self.L_states
                    squared_dist = np.sum(np.square(diff), axis=1)
                    safe = squared_dist <= self.state_squaraed_dist_u

                    if np.sum(safe) > 0:
                        at_boundary = False
                    else:
                        at_boundary = True

            if not at_boundary:
                # If for any state, the boundary condition is now fulfilled, remove it from the list.
                print("Removed",self.Failed_experiment_list[i],self.Failed_state_list[i])
                self.Failed_experiment_list.pop(i)
                self.Failed_state_list.pop(i)







    def select_gp_subset(self):
        '''
        Select subset of data for the full GP. As we do not expand in the safe set, random subset can be chosen.
        However, as we would like good inference with less uncertainty for points with small lower_bound (more likely that we don't hit the interior of our safe set),
        we sample at random based on a probability proportial to the lower bounds.
        '''
        # Determine slack for all the points in the dataset
        constraint_idx = np.where(self.fmin != -np.inf)
        constraint_idx=np.asarray(constraint_idx).reshape(-1,1)
        # Determine slack
        slack=self.lower_bound[:,constraint_idx]-self.fmin[constraint_idx]-self.eta_L*self.scaling[constraint_idx]
        if len(constraint_idx)>1:
            slack=np.min(slack,axis=1)
        slack=slack.squeeze()
        # Consider all points other than the ones for x_0.
        idx_rest = [x for x in list(range(self._x.shape[0])) if x not in self.x_0_idx_full_data]
        # Determine probability proportional to the lower bound and sample based on it
        slack_rest=slack[idx_rest]
        dist=np.exp(-5*(slack_rest**2))
        prob=dist/np.sum(dist)
        random_id = np.random.choice(idx_rest,size=self.N_reset, p=prob,replace=False)
        # Sample points and include all data points corresponding to x_0.
        X_full=self._x[random_id,:]
        X_full=np.vstack((self._x[self.x_0_idx_full_data,:],X_full))
        Y_full = self._y[random_id, :]
        Y_full = Y_full.squeeze()
        Y_full = np.vstack((self._y[self.x_0_idx_full_data, :], Y_full))
        self.in_gp_full[:]= False
        self.in_gp_full[self.x_0_idx_full_data] = True
        self.in_gp_full[random_id]=True
        self.x_0_idx_gp_full=np.arange(start=0,stop=self._x[self.x_0_idx_full_data,:].shape[0],step=1)
        # set GP points
        for i, gp in enumerate(self.gp_full):
            gp.set_XY(X_full, Y_full[:, i].reshape(-1, 1))

    def find_constraint_max(self, a_init, x, iters=None):
        """
        Swarm optimization used for calculating the best safe action for a give state x

        Parameters
        ----------
        a_init: ndarray
            Actions with which we initialize our swarm

        x: ndarray
            State for which we want to find the safe action

        iters: int
            Number of iterations to run for swarm optimization
            If None, max_iters from the class is used

        Returns
        -------
        global_best: Best action which optimizes the swarm

        """
        iterations = iters
        if iters is None:
            iterations = self.max_iters

        # Define the objective function for the swarm
        constraint_objective = lambda a: self.constraint_objective(a, x=x)
        bounds = np.asarray(self.bounds)
        # Define the swarm
        swarm_maximizer = SwarmOptimization(self.swarm_size, self.optimal_velocities,
                                            constraint_objective,
                                            bounds=bounds)

        # Pick particles for swarm initialization
        random_id = np.random.randint(a_init.shape[0], size=self.swarm_size)
        a = a_init[random_id, :]
        # perturb particles if desired
        if self.perturb_particles:
            u = np.random.uniform(-1, 1, size=a.shape)
            a = a + u * self.optimal_velocities
            bound = np.array(self.bounds).T
            a = np.clip(a, bound[0], bound[1])

        # Run swarm and find the global best
        swarm_maximizer.init_swarm(a)
        swarm_maximizer.run_swarm(iterations)
        global_best = swarm_maximizer.global_best[None, :]
        return global_best

    def constraint_objective(self, a, x):

        """
        Objective used for the swarm optimization which finds the best action for a given state

        Parameters
        ---------
        a: ndarray
            action for which we evaluate the objective

        x: ndaarray
            state for which we want to find the best action

        Returns
        -------
        lower_bound: ndarray
            minimum of the lower_bounds for each particle over all constraint functions

        global_safe: ndarray
            Boolean array indicating if the particle is safe (true for all particles here)
        """

        # stack up the particles
        particles = np.hstack((a, np.tile(x, (a.shape[0], 1))))
        lower_bound = np.ones(a.shape[0]) * np.inf
        beta = self.beta(self.t)
        global_safe = np.ones(a.shape[0], dtype=bool)
        # Loop over each GP to find the minimum slack
        for i, (gp, scaling) in enumerate(zip(self.gp_full, self.scaling)):
            if self.fmin[i] == -np.inf:
                continue

            mean, var = gp.predict_noiseless(particles)
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())
            lb = mean - beta * std_dev
            slack = np.atleast_1d(lb - self.fmin[i])
            lower_bound = np.minimum(lower_bound, slack / scaling)

        return lower_bound, global_safe




















