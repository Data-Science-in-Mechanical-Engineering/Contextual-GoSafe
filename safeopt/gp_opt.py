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


__all__ = ['SafeOpt', 'SafeOptSwarm',"GoSafe"]


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
                 swarm_size=20):
        """Initialization, see `SafeOptSwarm`."""
        super(SafeOptSwarm, self).__init__(gp,
                                           fmin=fmin,
                                           beta=beta,
                                           num_contexts=0,
                                           threshold=threshold,
                                           scaling=scaling)

        # Safe set
        self.S = np.asarray(self.gps[0].X)

        self.swarm_size = swarm_size
        self.max_iters = 100  # number of swarm iterations

        if not isinstance(bounds, list):
            self.bounds = [bounds] * self.S.shape[1]
        else:
            self.bounds = bounds

        # These are estimates of the best lower bound, and its location
        self.best_lower_bound = -np.inf
        self.greedy_point = self.S[0, :]

        self.optimal_velocities = self.optimize_particle_velocity()

        swarm_types = ['greedy', 'maximizers', 'expanders']

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





class GoSafe(SafeOpt):
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
        List of parameters, includes actions and initial conditions
    fmin: list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted)
    x_0: Numpy array
        Initial condition we would like to optimize for
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

     extensions: Boolean
         Used to indicate if extended version of GoSafe should be used, if False
         the standard GoSafe implementation is used.

    Constraint_bounds: Boolean
        Constraints lower_bound l_n = max(l_{n-1},mu-beta*sigma), u_n = max(u_{n-1},mu+beta*sigma)

    Examples
    --------
    >>> from safeopt import GoSafe
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

    >>> opt = GoSafe(gp, parameter_set, fmin=[0.],x_0=np.array([1,1]))
    >>> next_parameters = opt.optimize()

    Add a new data point with the parameters and the performance to the GP. The
    performance has normally be determined through an external function call.

    >>> performance = np.array([[1.]])
    >>> opt.add_new_data_point(next_parameters, performance)
    """

    def __init__(self, gp, parameter_set, fmin,x_0, beta=2,
                 num_contexts=0, threshold=0, scaling='auto',extensions=False,constraint_bounds=False,eps=0.1):
        """Initialization, see `SafeOpt`."""
        super(GoSafe, self).__init__(gp,
                                      parameter_set=parameter_set,
                                      fmin=fmin,
                                      lipschitz=None,
                                      beta=beta,
                                      num_contexts=num_contexts,
                                      threshold=threshold,
                                      scaling=scaling)

        # Dimension of state
        state_dim=np.shape(x_0)[0]
        #Indexes where states are stored in the input vector
        self.state_idx=list(range(parameter_set.shape[1]-state_dim,parameter_set.shape[1]))
        #All combinations with initial condition x_0
        self.x_0_idx =np.sum(parameter_set[:,self.state_idx]==x_0,axis=1)==state_dim
        if not np.any(self.x_0_idx):
            raise AssertionError("Initial state x_0 not in parameter set")



        #Only need M for IC x_0
        self.M = self.S[self.x_0_idx].copy()
        self.safe_states=self.S.copy()

        #Used to make modifications on GoSafe, currently unimplemented
        self.extensions=extensions
        self.x_0 = x_0
        self.constraint_bounds=constraint_bounds
        #Using as an indicator to note that Q has been initialized arbitrarily, will be required for constraining
        self.Q_empty=True
        self.criterion="S1" #Which step we are running at the moment
        self.eps=eps #Defined to switch between S1,S2,S3


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


        if self.constraint_bounds and not self.Q_empty: #If Q has been updated once and we want to constrain our bounds
            # Iterate over all functions
            for i in range(len(self.gps)):
                # Evaluate acquisition function
                mean, var = self.gps[i].predict_noiseless(self.inputs)

                mean = mean.squeeze()
                std_dev = np.sqrt(var.squeeze())

                # Update confidence intervals
                self.Q[:, 2 * i] = np.max(self.Q[:, 2 * i] ,mean - beta * std_dev)
                self.Q[:, 2 * i + 1] = np.min(self.Q[:, 2 * i + 1] ,mean + beta * std_dev)

        else:
            # Iterate over all functions
            for i in range(len(self.gps)):
                # Evaluate acquisition function
                mean, var = self.gps[i].predict_noiseless(self.inputs)

                mean = mean.squeeze()
                std_dev = np.sqrt(var.squeeze())

                # Update confidence intervals
                self.Q[:, 2 * i] = mean - beta * std_dev
                self.Q[:, 2 * i + 1] = mean + beta * std_dev

                self.Q_empty=False

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

        # Reference to confidence intervals, gets l(a,x,0), u(a,x,0)
        l, u = self.Q[:, :2].T

        if not np.any(self.S):
            self.M[:] = False
            self.G[:] = False
            return

        #Lower bound and upper bound for initial condition IC
        u_x0=u[self.x_0_idx]
        l_x0=l[self.x_0_idx]
        # Set of possible maximisers
        # Maximizers: safe upper bound above best, safe lower bound
        #Find pairs (x_0,a) which lie in S_n
        safe_x0 = self.S[self.x_0_idx]
        self.M[:] = False
        self.M[safe_x0] = u_x0[safe_x0] >= np.max(l_x0[safe_x0])
        max_var = np.max(u_x0[self.M] - l_x0[self.M]) / self.scaling[0]

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

        #Start with S1

        if full_sets:
            s = self.x_0_idx
            var_G=np.max((u[s, :] - l[s, :]) / self.scaling,axis=1)

            do_S1=np.max(np.max(var_G),max_var)>= self.eps

        else:
            # skip points in M, they will already be evaluated for x_0
            s=np.zeros(self.inputs.shape[0], dtype=np.bool)
            s[self.x_0_idx]= np.logical_and(safe_x0, ~self.M) #Sn,x0 \Mn
            # Remove points with a variance that is too small
            s[s] = (np.max((u[s, :] - l[s, :]) / self.scaling, axis=1) >
                    max_var)
            s[s] = np.any(u[s, :] - l[s, :] > self.threshold*beta, axis=1) #If any point has a minimal variance of thresshold

            #Check if we should apply S1
            if not np.any(s):
                #If we have no candidates for G but we are uncertain about some points in M_n, apply S1
                if max_var >= self.eps:
                    self.criterion = "S1"
                    return
                else:
                    do_S1=False
            #If we have candidates for G
            else:
                #Check if we are uncertain about any of our candidates, if yes apply S1
                do_S1=np.any((np.max((u[s, :] - l[s, :]) / self.scaling,axis=1) )>= self.eps)
                # no need to evaluate any points as expanders in G, exit
             #   return

        def sort_generator(array):
            """Return the sorted array, largest element first."""
            return array.argsort()[::-1]

        #If we apply S1
        if do_S1:
            #Update criterion for querying point
            self.criterion = "S1"

            # set of safe expanders
            G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

            if not full_sets:
                # Sort, element with largest variance first for fixed x_0 (maximum is taken along gp's dim)
                sort_index = sort_generator(np.max(u[s, :] - l[s, :],
                                                   axis=1))
            else:
                # Sort index is just an enumeration of all safe states
                sort_index = range(len(G_safe))

            # Get all unsafe points with initial state x_0
            unsafe_points = np.logical_not(self.S)  # All unsafe points
            unsafe_points = np.logical_and(unsafe_points, self.x_0_idx)  # All unsafe points for x_0


            for index in sort_index:
                # if self.use_lipschitz:
                #     # Distance between current index point and all other unsafe
                #     # points
                #     d = cdist(self.inputs[s, :][[index], :],
                #               self.inputs[~self.S, :])
                #
                #     # Check if expander for all GPs
                #     for i in range(len(self.gps)):
                #         # Skip evaluation if 'no' safety constraint
                #         if self.fmin[i] == -np.inf:
                #             continue
                #         # Safety: u - L * d >= fmin
                #         G_safe[index] =\
                #             np.any(u[s, i][index] - self.liptschitz[i] * d >=
                #                    self.fmin[i])
                #         # Stop evaluating if not expander according to one
                #         # safety constraint
                #         if not G_safe[index]:
                #             break
                # else:
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

                    mean2, var2 = gp.predict_noiseless(self.inputs[unsafe_points])

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
            del unsafe_points

            do_S1 = np.any((np.max((u[self.G, :] - l[self.G, :]) / self.scaling, axis=1)) >= self.eps)

            if do_S1:
                return

        elif not do_S1: #If we do not do S1, we do S2 or S3

            # Get the indexes corresponding to constraints as in S2 we are only considering expansions
            num_constraints = len(self.gps)
            # Check if performance function is constrained, if not look at uncertainty from 1:num_constraints
            start_constraint = (self.fmin[0] == None) * 1

            self.G[:] = False
            self.M[:] = False
            s = np.logical_not(self.x_0_idx) #Consider all IC which are not x_0
            #Check if we should do S2: the variance for any potential query point is greater than epsilon
            var_G = np.max((u[s, start_constraint:num_constraints] - l[s, start_constraint:num_constraints]) / self.scaling, axis=1)
            do_S2=np.any(var_G>=self.eps)



            if do_S2:
                self.criterion="S2"
                G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

                if not full_sets:
                    sort_index = sort_generator(np.max(u[s, :] - l[s, :],
                                                       axis=1))
                else:
                    sort_index = range(len(G_safe))

                for index in sort_index:
                    # if self.use_lipschitz:
                    #     # Distance between current index point and all other unsafe
                    #     # points
                    #     d = cdist(self.inputs[s, :][[index], :],
                    #               self.inputs[~self.S, :])
                    #
                    #     # Check if expander for all GPs
                    #     for i in range(len(self.gps)):
                    #         # Skip evaluation if 'no' safety constraint
                    #         if self.fmin[i] == -np.inf:
                    #             continue
                    #         # Safety: u - L * d >= fmin
                    #         G_safe[index] =\
                    #             np.any(u[s, i][index] - self.liptschitz[i] * d >=
                    #                    self.fmin[i])
                    #         # Stop evaluating if not expander according to one
                    #         # safety constraint
                    #         if not G_safe[index]:
                    #             break
                    # else:
                        # Check if expander for all GPs (only consider the constraints)
                    for i, gp in enumerate(self.gps[start_constraint:num_constraints]):
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
                #Check for the set G if we are uncertain about any point, if yes do S2.
                do_S2 = np.any((np.max((u[self.G, start_constraint:num_constraints] - l[self.G, start_constraint:num_constraints]) / self.scaling, axis=1)) >= self.eps)
                if do_S2:
                    return

            #S3
            elif not do_S2:
                self.G[:] = False
                self.M[:] = False
                self.criterion="S3"

                #Function defined to get indexes of all safe sets in the parameter set.
                ## states represents a unique vector of all distinct safe states
                def get_safe_state_idx(states):
                    safe_state_idx = np.zeros(self.inputs.shape[0], dtype=np.bool)
                    #Loop over all unique states
                    for state in states:
                        #Check which indices are associated with safe states
                        idx = np.where(np.sum(self.inputs[self.S, self.state_idx] == state,axis=1)==len(self.state_idx))
                        safe_state_idx[idx] = True

                    #Vector returns all parameter combinations associated with safe states
                    return safe_state_idx

                    # Find all states for which we have a safe action
                #Get all states that are unique in the Safe set
                safe_states=self.inputs[self.S, :]
                safe_states = np.unique(safe_states[:,self.state_idx],axis=0)
                self.safe_states = get_safe_state_idx(safe_states)




    def get_new_query_point(self, ucb=False):
        """
        Compute a new point at which to ealuate the function.

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

            if self.criterion=="S1":
                MG=np.zeros(self.inputs.shape[0], dtype=np.bool)#All false
                MG[self.x_0_idx] = np.logical_or(self.M, self.G[self.x_0_idx])
                value = np.max((u[MG] - l[MG]) / self.scaling, axis=1)
                x = self.inputs[MG, :][np.argmax(value), :]

                del MG

            elif self.criterion=="S2":
                #Only look at uncertainty for constraints
                num_constraints=len(self.gps)
                #Check if performance function is constrained, if not look at uncertainty from 1:num_constraints
                start_constraint=(self.fmin[0]==None)*1
                value = np.max((u[self.G,start_constraint:num_constraints] - l[self.G,start_constraint:num_constraints]) / self.scaling, axis=1)
                x = self.inputs[self.G, :][np.argmax(value), :]


            else:
                sampling_options=np.logical_and(self.safe_states,~self.S)
                # Only look at uncertainty for constraints
                num_constraints = len(self.gps)
                # Check if performance function is constrained, if not look at uncertainty from 1:num_constraints
                start_constraint = (self.fmin[0] == None) * 1
                value = np.max((u[sampling_options, start_constraint:num_constraints] - l[sampling_options, start_constraint:num_constraints]) / self.scaling, axis=1)
                x = self.inputs[sampling_options, :][np.argmax(value), :]

        if self.num_contexts:
            return x[:-self.num_contexts]
        else:
            return x

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
        if not np.any(self.S[self.x_0_idx]):
            return None

        #Only look at lower bounds for (a,x_0) in the safe set
        l = self.Q[self.x_0_idx, 0][self.S[self.x_0_idx]]

        max_id = np.argmax(l)
        # Remove all contexts from the parameter space
        param_without_context = self.inputs[self.x_0_idx, :]
        param_without_context=param_without_context[self.S[self.x_0_idx], :][max_id, :-self.num_contexts or None]

        #Return only the action and not the state
        return (param_without_context[:-len(self.state_idx)],
                l[max_id])













