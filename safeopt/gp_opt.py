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
from scipy.spatial import ConvexHull
from .utilities import (plot_2d_gp, plot_3d_gp, plot_contour_gp,
                        linearly_spaced_combinations)
from .swarm import SwarmOptimization


import logging


__all__ = ['SafeOpt', 'SafeOptSwarm',"GoSafe","GoSafeSwarm","GoSafeSwarm_Contextual"]



def unique(array):
    uniq, index = np.unique(array, return_index=True,axis=0)
    return uniq[index.argsort(),:]
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


        if self.fmin[0]==-np.inf:
            # Select the minimal velocity (for the toughest safety constraint)
            velocities = np.min(velocities[1:,:], axis=0)
        else:
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

    This class implements the `GoSafe` algorithm. It uses a Gaussian
    process model in order to determine parameter combinations that are safe
    with high probability. Based on these, it aims to both expand the set of
    safe parameters and to find the optimal parameters within the safe set.
    Furthermore, unlike safeopt, gosafe is able to discover disconnected safe sets.

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

    eps: double
        Used as stopping criteria for running S1 and S2 (as in paper)

    tol: double
        tolerance, multiplied with the variance to check for boundary conditions

    eta: double
        tolerance, added to tol*sigma to check for boundary condition

    max_ic_expansions: int
        maximum number of S2 and S3 steps performed. Once reached, we only sample unsafe actions for x_0 -> S3_IC criteria

    Examples
    --------
    >>> from safeopt import GoSafe
    >>> from safeopt import linearly_spaced_combinations
    >>> import GPy
    >>> import numpy as np

    Define a Gaussian process prior over the performance
    >>> x_0=[1, 1]
    >>> x = np.array([[0.,x_0[0],x_0[1]]])
    >>> y = np.array([[1.]])
    >>> gp = GPy.models.GPRegression(x, y, noise_var=0.01**2)

    >>> bounds = [[-1., 1.],[-1., 1.],[-1., 1.]]
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

    def __init__(self, gp, parameter_set, fmin,x_0,lipschitz=None ,beta=2,
                 num_contexts=0, threshold=0, scaling='auto',extensions=False,constraint_bounds=False,eps=0.1,tol=0.5,eta=0.2,max_ic_expansions=100000):
        """Initialization, see `SafeOpt`."""
        super(GoSafe, self).__init__(gp,
                                      parameter_set=parameter_set,
                                      fmin=fmin,
                                      lipschitz=lipschitz,
                                      beta=beta,
                                      num_contexts=num_contexts,
                                      threshold=threshold,
                                      scaling=scaling)

        # Dimension of state
        self.state_dim=np.shape(x_0)[0]
        # Indexes where states are stored in the input vector
        self.state_idx=list(range(parameter_set.shape[1]-self.state_dim,parameter_set.shape[1]))


        # In general, x_0 may not lie in the discretized state space, look for the closest state (in L2 sense) in that case
        self.unique_states=np.unique(parameter_set[:,self.state_idx],axis=0)
        closest_state_idx=np.argmin(np.linalg.norm(self.unique_states-x_0,axis=1))
        closest_state=self.unique_states[closest_state_idx]
        # All combinations with initial condition x_0
        self.x_0_idx =np.sum(parameter_set[:,self.state_idx]==closest_state,axis=1)==self.state_dim


        #_boundary_data_points is a vector used to store whether each point in the GP hit the boundary or not and if it did for which state (which index)
        self._bounadry_data_points=np.ones([self._x.shape[0],1],dtype=int)*-1
        # Index of state which hit the boundary, if -1 then experiment was safe
        self._boundary_state_idx=-1
        # Only need M for IC x_0
        self.M = self.S[self.x_0_idx].copy()

        # Safe_states, a boolean vector that indicates which parameter combinations correspond to safe states
        self.S3_combinations=self.S.copy()

        # Used to make modifications on GoSafe, currently unimplemented
        self.extensions=extensions
        self.x_0 = x_0
        self.constraint_bounds=constraint_bounds
        #Using as an indicator to note that Q has been initialized arbitrarily, will be required for constraining
        self.Q_empty=True
        self.criterion="S1" # Which step we are running at the moment -> [S1,S2,S3,S3_IC], where S3_IC is when we run the step S3 and fix the state to be IC
        self.eps=eps # Defined to switch between S1,S2,S3
        self.expanding_steps=0 # Number of steps we did S2 and S3
        self.max_ic_expansions=max_ic_expansions # Maximum number of expansions we do for an arbitrary IC condition.
        # Tolerance used for boundary constraints
        self.tol=tol
        self.eta=eta

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

    def compute_unsafe_expansion_set(self):

        """
            Compute unsafe state action pairs for safe states
        """
        # Find all states for which we have a safe action

        # Get all states that are unique in the Safe set
        safe_options = self.inputs[self.S, :]
        safe_states = np.unique(safe_options[:, self.state_idx], axis=0)
        # Get all unsafe parameter combinations
        unsafe_combinations=np.logical_not(self.S)
        unsafe_inputs=self.inputs[unsafe_combinations, :]

        #Collect all rollouts that failed
        failed_data_idx = np.where(self._bounadry_data_points != -1)[0]

        failed_pair=self._x[failed_data_idx]

        for i in range(failed_pair.shape[0]):
            idx=np.where(np.sum(unsafe_inputs==failed_pair[i],axis=1)==failed_pair.shape[1])
            unsafe_combinations[idx]=False


        unsafe_inputs = self.inputs[unsafe_combinations, :]
        combinations_idx=np.zeros(np.sum(unsafe_combinations),dtype=bool)

        # Loop over all unique states to check unsafe parameter combinations associated with the safe states
        for i in range(safe_states.shape[0]):
            # Get all indexes in the unsafe parameter set which correspond to safe_states[i]
            idx = np.where(np.sum(unsafe_inputs[:, self.state_idx] == safe_states[i], axis=1) == self.state_dim)[0]
            # For all unsafe combination for safe state i, set idx to true
            combinations_idx[idx]= True
        #The updated combinations are used for S3
        self.S3_combinations[unsafe_combinations]=combinations_idx

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

        if not np.any(self.S):
            self.M[:] = False
            self.G[:] = False
            self.criterion = "S1"
            return

        # Get the indexes corresponding to constraints
        num_constraints = len(self.gps)
        # Check if performance function is constrained, if not look at uncertainty from 1:num_constraints
        start_constraint = (self.fmin[0] == -np.inf) * 1

        safe_x0 = self.S[self.x_0_idx] # All indexes corresponding to (a,x_0) in S_n

        # l,u for all i in I
        l = self.Q[:, ::2]
        u = self.Q[:, 1::2]


        # Get l(a,x,0), u(a,x,0) for our initial condition IC
        l_x0, u_x0 = self.Q[self.x_0_idx, :2].T

        # Set of possible maximisers
        # Maximizers: safe upper bound above best, safe lower bound
        self.M[:] = False
        self.M[safe_x0] = u_x0[safe_x0] >= np.max(l_x0[safe_x0])
        max_var = np.max(u_x0[self.M] - l_x0[self.M]) / self.scaling[0]




        self.G[:] = False

        #Reset all combinations for S3 (need to be revaluated)
        self.S3_combinations[:] = False

        # For the run of the algorithm we do not need to calculate the
        # full set of potential expanders:
        # We can skip the ones already in M and ones that have lower
        # variance than the maximum variance in M, max_var or the threshold.
        # Amongst the remaining ones we only need to find the
        # potential expander with maximum variance

        #Start with S1

        if full_sets:
            s = np.zeros(self.inputs.shape[0], dtype=np.bool)
            s[self.x_0_idx]=safe_x0
            var_G=np.max((u[s, :] - l[s, :]) / self.scaling,axis=1)

            do_S1=np.any(var_G > self.eps)

        else:
            # skip points in M, they will already be evaluated for x_0
            s=np.zeros(self.inputs.shape[0], dtype=np.bool)
            s[self.x_0_idx]= np.logical_and(safe_x0, ~self.M) #Sn,x0 \Mn
            # Remove points with a variance that is too small
            s[s] = (np.max((u[s, :] - l[s, :]) / self.scaling, axis=1) >
                    max_var)
            s[s] = np.any(u[s, :] - l[s, :] > self.threshold*beta, axis=1) #If any point has a minimal variance of thresshold

            # Check if we should apply S1
            if not np.any(s):
                # If we have no candidates for G but we are uncertain about some points in M_n, apply S1
                if max_var > self.eps:
                    self.criterion = "S1"
                    return
                else:
                    do_S1=False
            else:
                do_S1=np.any(np.max((u[s, :] - l[s, :]) / self.scaling, axis=1) >
                 self.eps)

        if do_S1:
            # Update criterion for querying point
            self.criterion="S1"
            def sort_generator(array):
                """Return the sorted array, largest element first."""
                return array.argsort()[::-1]

            # set of safe expanders
            G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

            if not full_sets:
                # Sort, element with largest variance first for fixed x_0 (maximum is taken along gp's dim)
                sort_index = sort_generator(np.max((u[s, :] - l[s, :])/self.scaling,
                                                   axis=1))
            else:
                # Sort index is just an enumeration of all safe states
                sort_index = range(len(G_safe))

            # Get all unsafe points with initial state x_0
            unsafe_points = np.logical_and(~self.S, self.x_0_idx)  # All unsafe points for x_0


            for index in sort_index:

                if self.use_lipschitz:
                #     Distance between current index point and all other unsafe
                #     points
                    params=self.inputs[s, :][[index], :]
                    unsafe_params=self.inputs[unsafe_points,:]
                    d_x = cdist(params[:,self.state_idx],
                               unsafe_params[:,self.state_idx])

                    d_a = cdist(params[:,0:np.min(self.state_idx)],
                            unsafe_params[:,0:np.min(self.state_idx)])
                #     # Check if expander for all GPs
                    for i in range(len(self.gps)):
                #         # Skip evaluation if 'no' safety constraint
                         if self.fmin[i] == -np.inf:
                             continue
                         # Safety: u - L * d >= fmin

                         L_x=self.liptschitz[i][1]
                         L_a=self.liptschitz[i][0]
                         G_safe[index] =\
                             np.any(u[s, i][index] - L_x * d_x - L_a*d_a >=
                                    self.fmin[i])
                #         # Stop evaluating if not expander according to one
                #         # safety constraint
                         if not G_safe[index]:
                             break
                else:
                    # Check if expander for all GPs
                    for i, gp in enumerate(self.gps):
                        # Skip evlauation if 'no' safety constraint
                        if self.fmin[i] == -np.inf:
                            continue

                        # Add safe point with its optimistic value to the gp
                        self._add_data_point(gp=gp,
                                             x=self.parameter_set[s, :][index, :],
                                             y=u[s, i][index],
                                             context=self.context)

                        # Get l2 of all unsafe points with state x_0
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
                # uncertain element gets picked by GoSafe anyways, we can
                # stop after we found the first one
                if G_safe[index] and not full_sets:
                    break

            del unsafe_points

            self.G[s] = G_safe

            # Check if variance in G_n U M_n is greater than epsilon
            if max_var>self.eps:
                return
            # If variance in M_n is less than eps, G_n is our only hope
            else:
                # Check if we could find any expander
                if np.any(G_safe):

                    # If variance in G_n greater than eps --> do_S1
                    do_S1= np.any((np.max((u[self.G,:] - l[self.G, :]) / self.scaling, axis=1)) > self.eps)
                    if do_S1:
                        return
                #If the set of expanders is empty and uncertainty in M_n is less than eps. We go to S2,S3
                else:
                    do_S1=False

        # Go to S2,S3 if S1 has converged
        if not do_S1:


            # Check if we have expanded the set enough, if yes, go to S3_IC
            if self.expanding_steps < self.max_ic_expansions:


                self.G[:] = False
                self.M[:] = False

                s=self.S
                #s = np.logical_and(self.S,~self.x_0_idx) # Consider all IC which are not x_0 but safe

                # Check if we should do S2: the variance for any potential query point is greater than epsilon
                var_G = np.max((u[s, start_constraint:num_constraints] - l[s, start_constraint:num_constraints]) / self.scaling[start_constraint:num_constraints], axis=1)
                do_S2=np.any(var_G>self.eps)

                if do_S2:

                    def sort_generator(array):
                        """Return the sorted array, largest element first."""
                        return array.argsort()[::-1]

                    self.criterion="S2"
                    G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

                    if not full_sets:
                        #Sort based on uncertainty w_n
                        sort_index = sort_generator(np.max((u[s, start_constraint:num_constraints] - l[s,
                                                                                                     start_constraint:num_constraints])/self.scaling[start_constraint:num_constraints],
                                                                                                        axis=1))
                    else:
                        sort_index = range(len(G_safe))

                    for index in sort_index:
                         if self.use_lipschitz:
                        #     # Distance between current index point and all other unsafe
                        #     # points
                            params = self.inputs[s, :][[index], :]
                            unsafe_params = self.inputs[~self.S, :]
                            d_x = cdist(params[:, self.state_idx],
                                        unsafe_params[:, self.state_idx])

                            d_a = cdist(params[:, 0:np.min(self.state_idx)],
                                        unsafe_params[:, 0:np.min(self.state_idx)])
                        #
                        #     # Check if expander for all GPs
                            for i in range(len(self.gps)):
                        #         # Skip evaluation if 'no' safety constraint
                                 if self.fmin[i] == -np.inf:
                                     continue
                                 # Safety: u - L * d >= fmin
                                 L_x = self.liptschitz[i][1]
                                 L_a = self.liptschitz[i][0]
                                 G_safe[index] =\
                                     np.any(u[s, i][index] - L_x * d_x - L_a*d_a >=
                                            self.fmin[i])
                        #         # Stop evaluating if not expander according to one
                        #         # safety constraint
                                 if not G_safe[index]:
                                     break

                         else:
                            # Check if expander for all GPs (only consider the constraints)
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


                    if np.any(G_safe):
                        # Update safe set (if full_sets is False this is at most one point)
                        self.G[s] = G_safe
                        # Check for the set G if we are uncertain about any point, if yes do S2.
                        do_S2 = np.any((np.max((u[self.G, start_constraint:num_constraints] - l[self.G,
                                                                                              start_constraint:num_constraints]) / self.scaling[start_constraint:num_constraints],
                                                                                                axis=1)) > self.eps)
                        if do_S2:
                            return
                    else:
                        do_S2=False

                # S3
                if not do_S2:
                    self.G[:] = False
                    self.M[:] = False
                    self.criterion="S3"

                    self.compute_unsafe_expansion_set()

            else:
                self.G[:] = False
                self.M[:] = False
                # This criteria corresponds to the case where we do not expand our safe set for any arbitrary IC but only for x_0
                self.criterion="S3_IC"




    def get_new_query_point(self, ucb=False):
        """
        Compute a new point at which to evaluate the function based on the current criterion = [S1,S2,S3,S3_IC]

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
                MG=np.zeros(self.inputs.shape[0], dtype=np.bool)# All false
                # As S1 only considers IC _x0, we only sample combinations corresponding to our IC
                MG[self.x_0_idx] = np.logical_or(self.M, self.G[self.x_0_idx]) #M_n U G_n

                # Find max_a max_i w_n(a,x_0)
                value = np.max((u[MG] - l[MG]) / self.scaling, axis=1)
                x = self.inputs[MG, :][np.argmax(value), :]

                del MG

            elif self.criterion=="S2":
                self.expanding_steps += 1  # Increase expanding step counter
                # Only look at uncertainty for constraints
                num_constraints=len(self.gps)
                # Check if performance function is constrained, if not look at uncertainty from 1:num_constraints
                start_constraint=(self.fmin[0]==-np.inf)*1
                # Find max_(a,x_0') max_i w_n(a,x_0') for i in I_g
                value = np.max((u[self.G,start_constraint:num_constraints] - l[self.G,
                                                                             start_constraint:num_constraints]) / self.scaling[start_constraint:num_constraints],
                                                                                axis=1)
                x = self.inputs[self.G, :][np.argmax(value), :]


            elif self.criterion=="S3":
                self.expanding_steps += 1  # Increase expanding step counter

                # Only look at uncertainty for constraints
                num_constraints = len(self.gps)
                # Check if performance function is constrained, if not look at uncertainty from 1:num_constraints
                start_constraint = (self.fmin[0] == -np.inf) * 1
                # Find max_(a,x_0') max_i w_n(a,x_0') for i in I_g  and (a,x_0') not in S_n but x_0' in safe states
                value = np.max((u[self.S3_combinations, start_constraint:num_constraints] - l[self.S3_combinations,
                                                                                        start_constraint:num_constraints]) / self.scaling[start_constraint:num_constraints],
                                                                                            axis=1)
                x = self.inputs[self.S3_combinations, :][np.argmax(value), :]

            elif self.criterion=="S3_IC": # Criterion S3_IC -> Only samples random actions for x_0
                self.expanding_steps += 1
                #Look at all unsafe combinations for x_0
                sampling_options=np.logical_and(self.x_0_idx,~self.S)

                #Find all failed combinations for x_0
                failed_data_idx = np.where(self._bounadry_data_points != -1)[0]
                failed_pair = self._x[failed_data_idx]
                x_0=np.unique(self.inputs[self.x_0_idx,self.state_idx])
                if np.any(failed_pair):
                    idx=np.where(np.sum(failed_pair[:,self.state_idx]==x_0,axis=1)==self.state_dim)

                    failed_pair=failed_pair[idx]
                    #Find and remove all combinations that led to failure
                    unsafe_inputs=self.inputs[sampling_options,:]
                    unsafe_combinations=sampling_options[sampling_options]
                    for i in range(failed_pair.shape[0]):
                        idx = np.where(np.sum(unsafe_inputs == failed_pair[i], axis=1) == failed_pair.shape[1])
                        unsafe_combinations[idx] = False

                    sampling_options[sampling_options]=unsafe_combinations



                num_constraints = len(self.gps)
                # Check if performance function is constrained, if not look at uncertainty from 1:num_constraints
                start_constraint = (self.fmin[0] == -np.inf) * 1
                # Find max_(a,x_0) max_i w_n(a,x_0) for i in I_g  and (a,x_0)
                value = np.max((u[sampling_options, start_constraint:num_constraints] - l[sampling_options,
                                                                                        start_constraint:num_constraints]) / self.scaling[start_constraint:num_constraints],
                                                                                            axis=1)
                x = self.inputs[sampling_options, :][np.argmax(value), :]

            else: #Introduced for debuging purposes, as criterion has to be in [S1,S2,S3,S3_IC]
                raise AssertionError("Invalid criterion for querying")


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
        #idx=np.logical_and(self.S,self.x_0_idx)
        l = self.Q[self.x_0_idx, 0][self.S[self.x_0_idx]]
        #l=self.Q[idx,0]
        max_id = np.argmax(l)
        # Remove all contexts from the parameter space
        param_without_context = self.inputs[self.x_0_idx, :]
        param_without_context=param_without_context[self.S[self.x_0_idx], :][max_id, :-self.num_contexts or None]

        #Return only the action and not the state
        return (param_without_context[:-self.state_dim],
                l[max_id])


    def at_boundary(self,state,context=None):
        """""
            Check if the provided state lies at the boundary and if this is the case return a safe action

            Parameters
               ----------
               state: ndarray
                   A vector containing the current state

               context: ndarray
                   A vector containing the current context

               Returns
               -------
               at_boundary - bool
                   True if state lies at the boundary
               a - ndarray
                   Recommended action (None if the state does not lie at the boundary)

               Fail - bool
                   True if the state is out of the safe set

               Notes
               -----
               In general the state may not lie within the discretized parameter set
               Hence, the safe actions associated to the closest state in the parameter set (measured using L2 norm)
               are considered to be safe actions for the real state.

               Furthermore, for all states and data points that lie at the boundary self._boundary_state_idx stores
               the index of the state. This is then used to indicate whether a experiment failed or not
        """
        a=None
        at_boundary = True
        Fail=False
        beta = self.beta(self.t)
        num_constraints = len(self.gps)
        # Check if performance function is constrained, if not look at uncertainty from 1:num_constraints
        start_constraint = (self.fmin[0] == -np.inf) * 1

        # Find the closest state in the parameter set
        dist=np.linalg.norm(self.unique_states-state,axis=1)
        state_idx=np.where(dist==dist.min())[0]
        # Incase multiple states are equally close, just pick any one of them
        if np.shape(state_idx)[0]>1:
            state_idx=state_idx[0]
        closest_state=self.unique_states[state_idx]
        # Get all safe_states
        safe_params = self.inputs[self.S, :]  # all safe parameter combinations
        # Find the closest state according to the second norm
        safe_idx=np.where(np.sum(safe_params[:,self.state_idx]==closest_state,axis=1)==self.state_dim)[0]
        if not np.any(safe_idx):
            Fail=True
            print("State has no safe action in S_n, experiment failed")
            return at_boundary,a,Fail

        # else:
        # Define all backup policies we could take for our state
        input = safe_params[safe_idx, :]
        #input[:, self.state_idx] = state
        l2 = self.Q[:, ::2]
        l2=l2[self.S,:]
        l2 = l2[safe_idx, :]
        u2 = self.Q[:, 1::2]
        u2=u2[self.S,:]
        u2=u2[safe_idx,:]
        # Add contexts if provided
        if context is not None:
            input = self._add_context(input, context)

        #constraint_check = np.zeros([input.shape[0],1])
        # Loop over all constraints
        var2=u2-l2
        constraint_check = l2 >= self.fmin + self.tol * np.sqrt(var2) + self.eta * self.scaling

        # for i, gp in enumerate(self.gps):
        #
        #     if self.fmin[i] == -np.inf:
        #         continue
        #     # Get lowerbound for parameter choice
        #     #mean2, var2 = gp.predict_noiseless(input)
        #     #l2 = mean2 - beta * np.sqrt(var2)
        #     l2=l[:,i]
        #     var2=u[:,i]-l[:,i]
        #     # Check if the safe state-action pair keep us safe by a tolerance
        #     constraint_satisified = l2 >= self.fmin[i] + self.tol * np.sqrt(var2)+self.eta*self.scaling[i]
        #     constraint_check+=constraint_satisified*1
        #     # Break as soon as we are at boundary (we do not have the necessary tolerance for all constraints)
        #     if np.sum(constraint_satisified) == 0:  # No state action pair fulfils constraint with the given tolerance
        #         break


        if np.any(np.sum(constraint_check,axis=1)== num_constraints):  # If even one state action pair combination fulfilled all constraints with the provided tolerance
            at_boundary = False

        # Return a safe action (here the action which maximizes the lowerbound of the reward is returned)
        # and update boundary index
        if at_boundary:
            # For simplicity function value f for [x,a] is not calculated, rather value for [x_close,a] is used
            #l2=self.Q[self.S,0].T
            #l2=l2[safe_idx]
            #mean2, var2 = self.gps[0].predict_noiseless(input)
            #l2 = mean2 - beta * np.sqrt(var2)
            max_id = np.argmax(l2[:,0])

            # Remove all contexts from the parameter space
            param_without_context = input[max_id, :-self.num_contexts or None]
            a = param_without_context[:-self.state_dim]

            # Save the index, will be used when data is stored/added to GP
            self._boundary_state_idx=state_idx[0]
        return at_boundary,a,Fail





    def check_rollout(self,state,action,context=None):

        """
               Check if a state from the rollout lies at the boundary and if this is the case return a safe action

               Parameters
               ----------
               state: ndarray
                   A vector containing the current state

               action: ndarray
                   Current action which is being applied

               context: ndarray
                   A vector containing the current context

               Returns
               -------
               at_boundary - bool
                   True if state lies at the boundary
               a - ndarray
                   Recommended action

                Fail - bool
                    True if the state is out of the safe set

               Notes
               -----
               As all rollouts from S1,S2 are safe with high probability, we only check for the boundary
               for criterion S3, S3_IC.
        """

        at_boundary = False
        Fail=False
        if self.criterion == "S3" or self.criterion=="S3_IC":

            at_boundary,a,Fail=self.at_boundary(state,context)

        if not at_boundary:
            a=action

        return at_boundary,a,Fail


    def Update_gp_data(self):

        """""
        Checks if failed experiments would still fail after updating GPs and safe sets
        If yes, these experiments are removed from the GP
        
        """""

        num_constraints = len(self.gps)
        # Check which points led to failure (index!=-1)
        failed_data_idx=np.where(self._bounadry_data_points!=-1)
        # Check at which state the failure took place
        boundary_states_idx=np.unique(self._bounadry_data_points[failed_data_idx])
        states=self.unique_states[boundary_states_idx]
        # Loop over each of the states from which we switched to a safe policy
        safe_params = self.inputs[self.S, :]
        for j in range(states.shape[0]):
            # Find the state in the safe set
            state_idx = np.sum(safe_params[:, self.state_idx] == states[j], axis=1) == self.state_dim

            # Collect the lower and upper bound to check if it would still lie at the boundary.
            l2=self.Q[self.S, ::2][state_idx,:]
            u2=self.Q[self.S,1::2][state_idx,:]
            constraint_check=l2 >= self.fmin + self.tol * np.sqrt(u2-l2) + self.eta*self.scaling

            # If the state is no more at the boundary, remove it from the GP
            if np.any(np.sum(constraint_check,axis=1)== num_constraints):
                # Find all data points in GP for which the state led to failure
                idx=np.where(self._bounadry_data_points==boundary_states_idx[j])[0]
                # Delete data points from the GP
                X = self.gps[0].X
                X = np.delete(X, idx, 0)
                self._x = np.delete(self._x, idx, 0)
                self._bounadry_data_points = np.delete(self._bounadry_data_points, idx, 0)
                # Loop over all y datapoints to delete them
                for gp in self.gps:
                    Y=gp.Y
                    Y=np.delete(Y,idx,0)
                    gp.set_XY(X, Y)

                self._y = np.delete(self._y, idx, 0)







    def add_new_data_point(self, x, y,context=None):
        """
        Add a new function observation to the GPs.

        This function is the same as for safeopt
        but it also monitors a _boundary_data_point
        vector which stores indexes of states at which
        an experiment failed. If experiment was successful
        stored index is -1

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
        # Store index of state at which experiment failed (-1 if experiment was successful)
        self._bounadry_data_points=np.concatenate((self._bounadry_data_points,np.array([[self._boundary_state_idx]])),axis=0)
        # Reset index counter
        self._boundary_state_idx=-1

    def remove_last_data_point(self):
        """Remove the data point that was last added to the GP. Updated to also remove the respective boundary point"""
        last_y = self._y[-1]

        for gp, yi in zip(self.gps, last_y):
            if not np.isnan(yi):
                gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])

        self._x = self._x[:-1, :]
        self._y = self._y[:-1, :]

        self._bounadry_data_points=self._bounadry_data_points[:-1,:]


class GoSafeSwarm(SafeOptSwarm):

    """GoSafe for larger dimensions using a Swarm Optimization heuristic.

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
    max_S1_steps: int
        Maximum number of consecutive S1 steps
    max_S2_steps: int
        Maximum number of consecutive S2 steps
    max_S3_steps: int
        Maximum number of consecutive S3 steps
    max_expansion_steps: int
        Maximum number of S1 and S2 steps we can perform
        without querying S3
    tol: double
        tolerance used for boundary condition
    eta: double
        Additional tolerance element used for the boundary conditon
    eps: double
        Maximum precision for convergence
    S3_x0_ratio: double
        Ratio of how many particles with initial condition x0 should
        be used in S3-> if 1, S3 only tries out actions for x0
    max_data_size: int
        Maximum number of data points of states other than x_0 that
        the GP can contain

    reset_size: int
        Number of data points we choose for our subset (only considering states
        other than x0).

    boundary_ratio: double
        Ratio of the boundary points selected in our subset


    Examples
    --------
    >>> from safeopt import GoSafeSwarm
    >>> import GPy
    >>> import numpy as np

    Define a Gaussian process prior over the performance
    >>> x_0=0.5
    >>> x = np.array([[0.,x_0]])
    >>> y = np.array([[1.]])
    >>> gp = GPy.models.GPRegression(x, y, noise_var=0.01**2)

    Initialize the Bayesian optimization and get new parameters to evaluate

    >>> opt = GoSafeSwarm(gp, fmin=[0.], bounds=[[-1., 1.],[-1., 1.]],x_0=np.asarray([[x_0]]))
    >>> next_parameters = opt.optimize()

    Add a new data point with the parameters and the performance to the GP. The
    performance has normally be determined through an external function call.

    >>> performance = np.array([[1.]])
    >>> opt.add_new_data_point(next_parameters, performance)

    """

    def __init__(self, gp, fmin, bounds,x_0,beta=2,scaling='auto', threshold=0,
                 swarm_size=20,max_S1_steps=30,max_S2_steps=10,max_S3_steps=30,max_expansion_steps=50,tol=0.3,eta=0.7,eps=0.1,S3_x0_ratio=0.7,max_data_size=400,reset_size=200,boundary_ratio=0.7):

        # Initialize all SafeOptSwarm params
        super(GoSafeSwarm, self).__init__(gp=gp,
                                          fmin=fmin,
                                          bounds=bounds,
                                          beta=beta,
                                          scaling=scaling,
                                          threshold=threshold,
                                          swarm_size=swarm_size,
                                          define_swarms=False)


        # Save state information
        self.state_dim=np.shape(x_0)[0]
        self.state_idx = list(range(self.gp.input_dim - self.state_dim, self.gp.input_dim))

        self.x_0=x_0.T
        # Update velocity for S1 (should only affect the actions)
        self.S1_velocities=self.optimal_velocities.copy()
        self.S1_velocities=self.S1_velocities[:-self.state_dim] #For S1, only look at parameters for the actions

        self.action_dim=self.gp.input_dim - self.state_dim
        ## Define the swarms
        bounds=np.asarray(self.bounds)
        # For S1, we need maximizers, expanders and greedy (used by maximizers)
        # As we only look at parameters in the action space, no velocity in
        # direction of the state is considered
        S1_swarm_types=['greedy', 'maximizers', 'expanders']
        self.swarms = {swarm_type:
            SwarmOptimization(
                swarm_size,
                self.S1_velocities,
                partial(self._compute_particle_fitness,
                        swarm_type),
                bounds=bounds[:self.action_dim].tolist())
            for swarm_type in S1_swarm_types}

        # For S2 we only consider expansions but can have velocities in all
        # directions (state and action)
        self.swarms['expanders_S2']=SwarmOptimization(swarm_size,
                                                     self.optimal_velocities,
                                                     partial(self._compute_particle_fitness,
                                                             'expanders_S2'),
                                                     bounds=bounds)

        # S3 only updates actions, need to ensure that starting state is safe. Furthermore, actions can be arbitary large (update velocity set)
        S3_velocities=self.optimal_velocities.copy()
        S3_velocities=S3_velocities[:-self.state_dim]
        S3_velocities=S3_velocities*10
        self.swarms['S3']=SwarmOptimization(swarm_size,
                                                     S3_velocities, #We can allow faster velocities for S3
                                                     partial(self._compute_particle_fitness,
                                                             'S3'),
                                                     bounds=bounds[:self.action_dim].tolist())

        # Criterion indicating which of the 3 methods S1,S2,S3 we are running atm
        self.criterion='init'
        # Boolean used to indicate if we should perturb particles for further exploration
        self.perturb_particles=True
        # Get where in the safe set x_0 is stored
        unique_safe_states=unique(self.S[:,self.state_idx])

        closest_state_idx = np.argmin(np.linalg.norm(unique_safe_states- self.x_0, axis=1))
        closest_state=unique_safe_states[closest_state_idx,:]
        # All combinations of x_0 in the safe set
        self.x_0_idx=np.where(np.sum(self.S[:,self.state_idx]==closest_state,axis=1)==self.state_dim)[0]
        # All combinations of x_0 in the GP
        self.x_0_idx_gp=self.x_0_idx.copy()
        # All combinations of x_0 in all the recorded data
        self.x_0_idx_gp_full_data=self.x_0_idx.copy()

        self.in_gp=np.ones(self.x_0_idx.shape[0],dtype=bool)
        # Get first estimate of the best action for the greed swarm
        self.greedy_point = self.S[self.x_0_idx[0], :-self.state_dim]

        # Set parameters for switching between S1,S2 and S3
        self.s1_steps=0
        self.s2_steps=0
        self.s3_steps=0
        self.max_S1_steps=max_S1_steps
        self.max_S2_steps=max_S2_steps
        self.max_S3_steps=max_S3_steps
        # switching from S3 to S1
        self.switch=False

        # Define tolerance parameters
        self.tol=tol
        self.eta=eta
        # Stores information from failed experiments (failed initial state action pair and the state at which we hit the boundary)
        self.Failed_experiment_list=[]
        self.Failed_state_list=[]
        # Fast distance approximate used to reduce computational cost while evaluating the covariances between points
        # If true, it only takes the kernel of f into consideration instead of all the GPs
        self.fast_distance_approximation=False

        # Set of states considered for S3, randomly sampled at the beginning and kept constant during swarm optimization
        self.S3_states=np.empty([self.swarm_size,self.state_dim])
        # Stores the set number each point in the safe set belongs to. Everytime S3 adds a new dada, current_set_number is updated and points
        # and it is encouraged to draw points from the current safe set -> Ensures we pursue new safe sets as soon as we get their info
        self.set_number=np.zeros(self.S.shape[0],dtype=int)
        # Ratio tells how many of the initial safe states for S3 should be x_0
        self.S3_x0_ratio=S3_x0_ratio
        # Current set number, used to define the region we are investigating
        self.current_set_number=0

        # epsilon parameter used for the convergence results
        self.eps=eps

        # Indicator if we should check full safe set and gp data for boundary condition
        self.check_full_data=True

        # Safety cutoff used in the boundary condition to check if we can apply an action of another
        # state x_c for our state x ( k(x,x_c) \geq 0.95, else experiment has failed)
        self.safety_cutoff=0.95

        # Variables defined for the subset selection schemes
        self.data_size_max = max_data_size
        self.N_reset=reset_size
        self.boundary_ratio=boundary_ratio

        # Lowerbounds for all points in the safe set
        self.lower_bound=np.ones([self.S.shape[0],len(self.gps)])*-np.inf

        self.inner_particles=np.zeros(self.S.shape[0],dtype=np.bool)
        self.eta_interior=1
        # Indicators used to upperbound S1 and S2 steps
        self.max_expansion_steps=max_expansion_steps
        self.expansion_steps=0

        self.use_convexhull=False
        self.encourage_jumps=True

    def _seed(self,seed=None):
        if seed is not None:
            np.random.seed(seed)

    def compute_particle_distance(self,particles,X_data,full=False):
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

                covariance_mat=np.ones([particles.shape[0],X_data.shape[0]])*np.inf

                for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                    # Do not consider GP if it has no constraint
                    if self.fmin[i] == -np.inf:
                        continue
                    # Take the pointwise minimum over each GP
                    covariance_mat = np.minimum(covariance_mat,gp.kern.K(particles, X_data)/scaling**2)

                return covariance_mat

            else:
                # Just return the covariance matrix of the objective
                covariance_mat=self.gp.kern.K(particles,X_data)
                covariance_mat /= self.scaling[0] ** 2
                return covariance_mat

        # Here we want to find the highest covariance between the particles
        # and points in X_data
        else:
            # fast_distance_approximation just considers the GP of the objective
            # If false, we take all GPs into account
            if not self.fast_distance_approximation:
                covariance=np.zeros(particles.shape[0])
                for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                    # Neglect point if no constraint
                    if self.fmin[i] == -np.inf:
                        continue
                    # Determine covariance
                    covariance_mat = gp.kern.K(particles,X_data.copy())
                    covariance_mat /=scaling**2
                    # Find maximum covariance for each particle
                    covariance_mat=np.max(covariance_mat,axis=1)

                    # Compare to the covariances we have so far and
                    # take the maximum
                    covariance=np.maximum(covariance,covariance_mat)

            else:
                # Determine covariance using solely the objective
                covariance = self.gp.kern.K(particles,X_data)
                covariance /= self.scaling[0] ** 2
                covariance = np.max(covariance, axis=1)
            return covariance



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
                * 'expander' : Expanders with x_0 fixed (lower bound close to constraint)
                * 'maximizer' : Maximizers (Upper bound better than best l)
                * 'expander_S2': Expanders but for all initial conditions
                * 'S3': Optimization in the full parameter space
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
        # Update particles to include the state for S1
        if swarm_type in ['greedy', 'maximizers', 'expanders']:
            # Add state to the particle
            particles=np.hstack((particles,np.tile(self.x_0,(particles.shape[0],1))))
        # Add state to the particle for S3
        if swarm_type == 'S3':
            particles=np.hstack((particles,self.S3_states.reshape(-1,self.state_dim)))

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

        is_safe = swarm_type == 'safe_set'
        is_expander = swarm_type == 'expanders'
        is_maximizer = swarm_type == 'maximizers'
        is_expander_S2 = swarm_type == 'expanders_S2'
        is_S3 = swarm_type == 'S3'
        # value we are optimizing for. Expanders and maximizers seek high
        if is_safe:
            self.lower_bound[:,0]=np.maximum(self.lower_bound[:,0],lower_bound)
            lower_bound=self.lower_bound[:,0]
        # variance points-> if the function has no constraints and we are doing S2
        # dont consider values for f in optimization
        if (is_expander_S2 or is_S3) and self.fmin[0]==-np.inf:
            values=np.zeros(len(std_dev))
        else:
            values = std_dev / self.scaling[0]




        if is_safe:
            interest_function = None
        else:
            if is_expander or is_expander_S2:
                # For expanders, the interest function is updated depending on
                # the lower bounds
                interest_function = (len(self.gps) *
                                     np.ones(np.shape(values), dtype=np.float))

                slack_min = np.ones(len(particles)) * np.inf

                #if is_expander_S2:
                #    interior_particles=self.S[self.inner_particles,:]
                #    covariance=self.compute_particle_distance(particles,interior_particles)
                #    interest_function *= np.exp(-5 * covariance)

            elif is_S3:
                interest_function = (len(self.gps) *
                                     np.ones(np.shape(values), dtype=np.float))

                # Determine covariance between particles and safe set

                # If we check full data we consider the safe set but also
                # all points collected so far during experiments
                if self.check_full_data:
                    X_data=np.vstack((self._x,self.S))

                else:
                    X_data=self.S

                # Get the maximum covariance between particles and X_data
                covariance = self.compute_particle_distance(particles,X_data)
                # Penalize high covariances -> Want to sample out of the safe set
                #interest_function*=np.exp(-5*covariance)

                # Check if some experiments have led to failure/hit boundary
                if self.Failed_experiment_list:
                    # Find covariance between between particles and failed experiments
                    failed_experiments=np.asarray(self.Failed_experiment_list).copy()
                    failed_experiments=failed_experiments.reshape(-1, particles.shape[1])
                    covariance_failedset = self.compute_particle_distance(particles,failed_experiments)

                    # Penalize points with high covariance -> Avoids sampling same points again

                    # Take the maximum between covariance with all the safe data and
                    # failed experiments
                    covariance=np.maximum(covariance,covariance_failedset)

                # Define interest function to encourage small covariances
                # between failedset and current safeset
                interest_function *= np.exp(-5 * covariance)


            elif is_maximizer:
                # Encourage points with u_n >= l_max
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
                if is_safe:
                    # Update lower bounds for points on the safe set as defined in the paper
                    # Using the definition of the contained set
                    self.lower_bound[:, i] = np.maximum(self.lower_bound[:, i], lower_bound)
                    lower_bound=self.lower_bound[:, i].copy()
                #upper_bound=mean+ beta * std_dev
                values = np.maximum(values, std_dev / scaling)

            # if the current GP has no safety constrain, we skip it
            if self.fmin[i] == -np.inf:
                continue
            # Dont care about safety of particles for S3
            if is_S3:
                continue
            slack = np.atleast_1d(lower_bound - self.fmin[i])
            #slack_up=np.atleast_1d(upper_bound - self.fmin[i])
            # computing penalties
            global_safe &= slack >= 0

            # Skip cost update for safety evaluation
            if is_safe:
                continue

            # Normalize the slack somewhat
            slack /= scaling

            total_penalty += self._compute_penalty(slack)

            if is_expander or is_expander_S2:
                slack_min=np.minimum(slack_min,slack)



        if is_expander or is_expander_S2:
            interest_function *= norm.pdf(slack_min, scale=0.2)


        # this swarm type is only interested in knowing whether the particles
        # are safe.
        if is_safe:
            self.inner_particles=np.min(self.lower_bound-self.fmin,axis=1)>=self.eta_interior
            return lower_bound, global_safe
        # Set particle values for particles with uncertainty less than epsilon to 0.
        # Avoids sampling particles which we are certain of
        ind = values < self.eps
        values[ind]=0
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
                * 'expanders_S2' : Finds expanders in the full state and action set
                * 'S3': Optimization over full

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
        # Get points from safe set for x_0
        safe_x0=self.S[self.x_0_idx]
        safe_x0_size=safe_x0.shape[0]
        num_x0_safe=safe[self.x_0_idx].sum()
        if num_x0_safe == 0:
            raise RuntimeError('The safe set for x0 is empty.')

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
            self.lower_bound=self.lower_bound[safe]
            self.inner_particles=self.inner_particles[safe]
            self.set_number=self.set_number[safe]
            self.x_0_idx=np.where(np.sum(self.S[:,self.state_idx]==self.x_0,axis=1)==self.state_dim)[0]
            safe_x0 = self.S[self.x_0_idx,:]
            safe_size = self.S.shape[0]
            safe_x0_size = safe_x0.shape[0]


        # initialize particles
        if swarm_type == 'greedy':
            # we pick particles u.a.r in the safe set for x_0

            random_id = np.random.randint(safe_x0_size, size=self.swarm_size - 3)
            best_sampled_point = np.argmax(self._y[self.x_0_idx_gp_full_data,0])
            x_0_actions=self._x[self.x_0_idx_gp_full_data]
            best_sampled_action=x_0_actions[best_sampled_point]
            last_x_0_point=x_0_actions[-1,:]
            # Particles are drawn at random from the safe set, but include the
            # - Previous greedy estimate
            # - last point
            # - best sampled point
            # Note: Particle only consist of safe actions for state x_0
            particles = np.vstack((safe_x0[random_id, :][:,:-self.state_dim],
                                   self.greedy_point,
                                   last_x_0_point[:-self.state_dim],
                                   best_sampled_action[:-self.state_dim]))
        else:
            # we pick particles u.a.r in the safe set
            # For S2:
            if swarm_type=="expanders_S2":
                # Find if we have points from the current_set_number which we can sample
                indexes=np.where(self.set_number == self.current_set_number)[0]
                if len(indexes)>0:
                    # If yes, sample only from the current set
                    random_id = np.random.choice(indexes, size=self.swarm_size)
                    # Points were sampled from the latest set
                    current_set = self.current_set_number
                else:
                    # Else sample from the previous safe set
                    current_set = np.max(self.set_number)
                    indexes=np.where(self.set_number == current_set)[0]
                    random_id = np.random.choice(indexes, size=self.swarm_size)


                particles = self.S[random_id, :]
                #if np.all(np.sum(particles[:,self.state_idx]==self.x_0,axis=1)==self.state_dim):
                if self.perturb_particles:
                    u=np.random.uniform(-1,1,size=particles.shape)
                    particles=particles+u*self.optimal_velocities
                    bounds=np.asarray(self.bounds)
                    bounds=bounds.T
                    particles = np.clip(particles, bounds[0], bounds[1])
            elif swarm_type=="S3":
                # Find how many points with IC x_0 should we use for S3
                size=int(self.swarm_size*self.S3_x0_ratio)
                # Sample remaining points
                idx_remaining=[x for x in list(range(safe_size)) if x not in self.x_0_idx]
                self.S3_states = np.zeros([self.swarm_size, self.state_dim])

                if len(idx_remaining)>0:
                    # Store all the sampled states
                    self.S3_states[:size,:]=self.x_0
                    if self.swarm_size - size > 0:
                        random_id = np.random.choice(idx_remaining, size=self.swarm_size - size)
                        self.S3_states[size:self.swarm_size,:] = self.S[random_id, self.state_idx].reshape(-1,self.state_dim)

                else:
                    self.S3_states[:,:]=self.x_0

                #self.S3_states = self.S3_states.reshape([self.swarm_size,self.state_dim])
                #action_idx=list(range(0,self.state_dim))
                # Take any random action
                bound=np.array(self.bounds)
                action=np.random.uniform(low=bound[:self.action_dim,0],high=bound[:self.action_dim,1],
                                         size=[self.swarm_size,self.action_dim])
                particles=action.reshape(-1,self.action_dim)
                current_set=self.current_set_number


            # For S1:
            else:
                # Check if for our x_0, we have any points in the newly discovered safe set
                indexes=np.where(self.set_number[self.x_0_idx]==self.current_set_number)[0]
                if len(indexes)>0:
                    # If yes, randomly sample from this set
                    random_id = np.random.choice(indexes, size=self.swarm_size)
                    current_set = self.current_set_number
                    particles = safe_x0[random_id, :][:, :-self.state_dim]
                else:
                    # Else, check if we have points for some other IC which has high covariance
                    # with our IC x_0
                    indexes = np.where(self.set_number == self.current_set_number)[0]
                    if len(indexes)>0:
                        # Calculated the maximum covariance between x_0 and other IC.
                        points=self.S[indexes,:]
                        points_x0=points.copy()
                        points_x0[:,self.state_idx]=self.x_0
                        covariance=self.compute_particle_distance(points_x0,points,full=True)
                        idx = np.argmax(covariance, axis=1)
                        covariance=np.max(covariance,axis=1)
                        # Check if covariance between these points and our IC
                        # is greater than the safety cutoff
                        idx_2=np.where(covariance>=self.safety_cutoff)[0]
                        if len(idx_2)>0:
                            # Randomly sample actions from these closeby safe sets
                            random_id = np.random.choice(idx_2, size=self.swarm_size)
                            particles=points[idx,:]
                            particles=particles[random_id,:]
                            particles=particles[:,:-self.state_dim]
                            current_set = self.current_set_number
                        else:
                            # If the cutoff condition is not fulfilled
                            # Find the latest safe set we have available for x_0
                            current_set = np.max(self.set_number[self.x_0_idx])
                            # Sample randomly from this safe set
                            indexes = np.where(self.set_number[self.x_0_idx] == current_set)[0]
                            random_id = np.random.choice(indexes, size=self.swarm_size)
                            particles = safe_x0[random_id, :][:, :-self.state_dim]
                    else:
                        # IF we do not have any points from the latest safe set
                        # Sample points from the most recent safe set available for
                        # x_0
                        current_set = np.max(self.set_number[self.x_0_idx])
                        indexes = np.where(self.set_number[self.x_0_idx] == current_set)[0]
                        random_id = np.random.choice(indexes, size=self.swarm_size)
                        particles = safe_x0[random_id, :][:, :-self.state_dim]

                if self.perturb_particles:
                    u = np.random.uniform(-1, 1, size=particles.shape)
                    particles = particles + u * self.S1_velocities
                    action_bounds=np.asarray(self.bounds[:-self.state_dim])
                    action_bounds=action_bounds.T #Lower upper bound are stored in each column
                    particles=np.clip(particles,action_bounds[0], action_bounds[1])

        # Run the swarm optimization
        swarm = self.swarms[swarm_type]
        swarm.init_swarm(particles)
        swarm.run_swarm(self.max_iters)

        # expand safe set
        if swarm_type != 'greedy' and swarm_type != 'S3':
            num_added = 0
            if swarm_type == 'maximizers' or swarm_type == 'expanders':

                # compute correlation between new candidates and current safe set
                size_best_pos=swarm.best_positions.shape[0]
                best_positions=np.hstack((swarm.best_positions,np.tile(self.x_0,(size_best_pos,1))))

                covariance = self.gp.kern.K(best_positions,
                                        np.vstack((self.S,
                                                   best_positions)))
            else:
                best_positions=swarm.best_positions
                covariance = self.gp.kern.K(best_positions,
                                            np.vstack((self.S,
                                                       best_positions)))
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
                    self.S = np.vstack((self.S, best_positions[[j], :]))
                    self.lower_bound=np.vstack((self.lower_bound,np.ones([1,len(self.gps)])*-np.inf))
                    self.inner_particles=np.hstack((self.inner_particles,0))
                    # Safe indicator of which set the point belongs to
                    self.set_number=np.vstack((self.set_number,current_set))
                    # Update index for x_0 if added point corresponds to x_0 IC
                    if np.sum(best_positions[[j],self.state_idx]==self.x_0)==self.state_dim:
                        # added point corresponds to initial condition x_0
                        self.x_0_idx=np.append(self.x_0_idx,[len(self.S)-1])

                    num_added += 1
                    mask[initial_safe + j] = True

            logging.debug("At the end of swarm {}, {} points were appended to"
                          " the safeset".format(swarm_type, num_added))

        elif swarm_type=='greedy':
            # check whether we found a better estimate of the lower bound
            greedy_point=np.hstack((self.greedy_point[None, :],self.x_0))
            mean, var = self.gp.predict_noiseless(greedy_point)
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())

            lower_bound = mean - beta * std_dev
            if lower_bound < np.max(swarm.best_values):
                self.greedy_point = swarm.global_best.copy()

        #if swarm_type == 'greedy':
            return swarm.global_best.copy(), np.max(swarm.best_values)

        # Find variances and best parameters for S3
        elif swarm_type=='S3':
            var = np.empty(len(self.gps), dtype=np.float)
            best_state_idx=np.argmax(swarm.best_values)
            state=np.asarray([self.S3_states[best_state_idx,:]])
            global_best=np.hstack((swarm.global_best[None, :],state))
            for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                var[i] = gp.predict_noiseless(global_best)[1]

            return global_best,np.sqrt(var)

        # compute the variance of the point picked
        var = np.empty(len(self.gps), dtype=np.float)
        # max_std_dev = 0.

        # Find variances and best parameters for S1,S2
        # S1
        if swarm_type == 'maximizers' or swarm_type == 'expanders':
            global_best=np.hstack((swarm.global_best[None, :],self.x_0))
        #S2
        else:
            global_best=swarm.global_best[None, :]
        for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
            var[i] = gp.predict_noiseless(global_best)[1]

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

        if self.gps[0].X.shape[0]-self.gps[0].X[self.x_0_idx_gp,:].shape[0]>=self.data_size_max:
            self.select_gp_subset()

        # If not actively expanding in the state space, run swarm to expand the safe set.
        if self.max_S2_steps == 0:
            if self.expansion_steps % 15 ==0:
                x_exp, std_exp = self.get_new_query_point('expanders_S2')

        expansion_steps=self.max_expansion_steps
        if self.encourage_jumps:
            indexes = np.where(self.set_number == self.current_set_number)[0]
            expansion_steps=np.minimum(20,self.max_expansion_steps)
            if len(indexes)>0:
                expansion_steps=self.max_expansion_steps
        # Check if we have exceeded maximum number of steps for S1, if yes go to S2
        if self.expansion_steps< expansion_steps:
            if self.s1_steps<self.max_S1_steps:
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
                if max(std_exp,std_maxi)>self.eps:
                    self.expansion_steps+=1
                    logging.info("The best maximizer has std. dev. %f" % std_maxi)
                    logging.info("The best expander has std. dev. %f" % std_exp)
                    logging.info("The greedy estimate of lower bound has value %f" %
                                 self.best_lower_bound)

                    if std_maxi >= std_exp:
                        return np.hstack((x_maxi.reshape(1,-1),self.x_0)).squeeze()#np.asarray([x_maxi[0],self.x_0[0]])
                    else:
                        return np.hstack((x_exp.reshape(1,-1),self.x_0)).squeeze()


            # Check if we have exceeded maximum number of S2 steps, if yes go to S3
            if self.s2_steps < self.max_S2_steps:

                #if self.s2_steps%10==0:
                #    if self.s2_steps%20==0:
                #        self.Reset_gp_data()
                #    self.Select_Gp_subset()
                self.s2_steps+=1
                #if self.s2_steps == self.max_S2_steps:
                 #   self.s1_steps=0
                self.criterion = "S2"
                x_exp, std_exp = self.get_new_query_point('expanders_S2')
                std_exp /= self.scaling
                std_exp = np.max(std_exp)
                # If uncertainty is less than eps, go to S3
                if std_exp>= self.eps:
                    self.expansion_steps += 1
                    logging.info("The best expander (S2) has std. dev. %f" % std_exp)
                    return x_exp.squeeze()







        self.criterion = "S3"
        # If self.s3_steps=0 -> We are running S3 for the first time after exploring the previous set,
        # Update set number
        if self.s3_steps==0:
            self.current_set_number+=1
        self.s3_steps += 1

        # Check if we have exceeded maximum number of S3 steps, if yes reset all step counters, if no sample from S3 swarm
        self.switch= self.s3_steps>=self.max_S3_steps
        if self.switch:
            self.s1_steps=0
            self.s2_steps = 0
            self.s3_steps=0
            self.expansion_steps=0
            self.switch = False



        x_exp, std_exp = self.get_new_query_point('S3')
        std_exp /= self.scaling
        std_exp = np.max(std_exp)
        logging.info("The best S3 point has std. dev. %f" % std_exp)
        return x_exp.squeeze()


    def get_maximum(self):
        """
        Return the current estimate for the maximum.

        Returns
        -------
        x : ndarray
            Location of the maximum
        y : ndarray
            Maximum value and constraint value

        """
        # Return the best point seen so far
        maxi = np.argmax(self._y[self.x_0_idx_gp_full_data,0])
        return self._x[self.x_0_idx_gp_full_data, :-self.state_dim][maxi,:], self._y[self.x_0_idx_gp_full_data, :][maxi,:]



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
        # If data point is added by S3, add it to the safe set
        if self.criterion=="S3":
            self.expansion_steps = 0
            # Add data point to the safe set
            # Maintain sparsity of safe set by only adding 1 point during S3
            self.criterion=="init"
            self.S = np.vstack((self.S, x))
            self.lower_bound=np.vstack((self.lower_bound,np.ones([1,len(self.gps)])*-np.inf))
            # Update step counters to run S1,S2 again
            self.s3_steps=0
            self.s1_steps=0
            self.s2_steps=0
            self.set_number = np.vstack((self.set_number, self.current_set_number))
            # If state for data point is x_0 -> x_0_idx counter
            if sum(x[0][self.state_idx] == self.x_0.squeeze())==self.state_dim:
                # added point corresponds to initial condition x_0
                self.x_0_idx = np.append(self.x_0_idx, [len(self.S) - 1])

            #self.s1_steps=0

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
        self.in_gp=np.append(self.in_gp,True)


        #if self.criterion=='S1':

        #    self.x_0_idx_gp = np.append(self.x_0_idx_gp, [self.gps[0].X.shape[0] - 1])

        # Check and update x_0_idx_gp counter
        if sum(x[0][self.state_idx]==self.x_0.squeeze())==self.state_dim:

            self.x_0_idx_gp = np.append(self.x_0_idx_gp, [self.gps[0].X.shape[0] - 1])
            self.x_0_idx_gp_full_data=np.append(self.x_0_idx_gp_full_data,[self._x.shape[0]-1])


    def add_data(self,x,y):
        # If data point is added by S3, add it to the safe set
        initial_size_gp=self.gps[0].X.shape[0]
        initial_size_full_data=self._x.shape[0]
        if self.criterion == "S3":
            new_point=x[0]
            self.expansion_steps = 0
            # Add data point to the safe set
            # Maintain sparsity of safe set by only adding 1 point during S3
            self.criterion == "init"
            self.S = np.vstack((self.S, new_point.reshape(1,-1)))
            self.lower_bound = np.vstack((self.lower_bound, np.ones([1, len(self.gps)]) * -np.inf))
            # Update step counters to run S1,S2 again
            self.s3_steps = 0
            self.s1_steps = 0
            self.s2_steps = 0
            self.set_number = np.vstack((self.set_number, self.current_set_number))
            # If state for data point is x_0 -> x_0_idx counter
            if sum(new_point[self.state_idx] == self.x_0.squeeze()) == self.state_dim:
                # added point corresponds to initial condition x_0
                self.x_0_idx = np.append(self.x_0_idx, [len(self.S) - 1])

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        if self.num_contexts:
            x = self._add_context(x, context)

        for i, gp in enumerate(self.gps):
            not_nan = ~np.isnan(y[:, i])
            if np.any(not_nan):
                # Add data to GP (context already included in x)
                gp.set_XY(np.vstack([gp.X, x[not_nan, :]]),
                          np.vstack([gp.Y, y[not_nan, i].reshape(-1,1)]))
                #self._add_data_point(gp, x[not_nan, :], y[not_nan, [i]])

        # Update global data stores
        self._x = np.concatenate((self._x, x), axis=0)
        self._y = np.concatenate((self._y, y), axis=0)

        add_to_data=np.ones(x.shape[0],dtype=bool)
        self.in_gp = np.append(self.in_gp, add_to_data)


        for i,new_point in enumerate(x):
            # Check and update x_0_idx_gp counter
            if sum(new_point[self.state_idx] == self.x_0.squeeze()) == self.state_dim:
                self.x_0_idx_gp = np.append(self.x_0_idx_gp, [initial_size_gp + i])
                self.x_0_idx_gp_full_data = np.append(self.x_0_idx_gp_full_data, [initial_size_full_data+i])



    def check_rollout(self,state,action):
        """
               Check if a state from the rollout lies at the boundary and if this is the case return a safe action

               Parameters
               ----------
               state: ndarray
                   A vector containing the current state

               action: ndarray
                   Current action which is being applied

               context: ndarray
                   A vector containing the current context

               Returns
               -------
               at_boundary - bool
                   True if state lies at the boundary

                Fail - bool
                    True if the state is out of the safe set

                a - ndarray
                   Recommended action

               Notes
               -----
                   As all rollouts from S1,S2 are safe with high probability, we only check for the boundary
                   for criterion S3
        """
        at_boundary=False
        Fail=False
        if self.criterion=="S3":
            # check if we are at the boundary, if yes return an alternative safe action
            at_boundary,Fail,a=self.at_boundary(state)

        if not at_boundary:
            a=action


        return at_boundary,Fail,a

    def at_boundary(self,state):
        """""
            Check if the provided state lies at the boundary and if this is the case return a safe action

            Parameters
               ----------
               state: ndarray
                   A vector containing the current state

           Returns
           -------
               at_boundary - bool
                   True if state lies at the boundary
               Fail - bool
                   True if the state is out of the safe set
               a - ndarray
                   Recommended action (None if the state does not lie at the boundary)

        """
        beta = self.beta(self.t)

        at_boundary=False
        Fail=False
        action = None
        # Check if we have any states in our safe set that are close to the state we are considering
        diff = np.abs(self.S[:, self.state_idx] - state)
        diff = (diff - self.optimal_velocities[self.state_idx]) <= 0
        idx = np.sum(diff, axis=1) == self.state_dim
        # If yes, use the constrained lower bounds to determine safety
        update_gp = False
        if np.sum(idx)>0:
            lower_bounds=self.lower_bound[idx,:]
            slack=lower_bounds-self.fmin-self.eta*self.scaling*1.2
            if np.any(np.sum(slack>0,axis=1) == len(self.gps)):
                return at_boundary,Fail,action

        # If check_full_data is true, uses all the data from GP to check for boundary condition
        if self.check_full_data:
            # Find the closest state in the full data
            full_data=self._x
            # Find all states such that they are closer than the optimal velocities
            # Computed for the swarms -> Have high covariance
            diff=np.abs(full_data[:,self.state_idx]-state)
            diff=(diff-self.optimal_velocities[self.state_idx])<=0
            idx=np.sum(diff,axis=1)==self.state_dim
            if np.sum(idx)==0:
                # If no states fulfil the velocity condition, find the closest state
                # in the L2 sense
                diff = np.linalg.norm(full_data[:, self.state_idx]- state, axis=1)
                closest_states_idx = np.where(diff == diff.min())[0]

                closest_states = full_data[closest_states_idx, :]
                closest_y = self._y[closest_states_idx, :]
                closest_states = closest_states.reshape(-1, np.shape(full_data)[1])
                # check how many of the closest states are in the GP
                in_gp = self.in_gp[closest_states_idx]
            else:
                closest_states = full_data[idx, :]
                closest_y = self._y[idx, :]
                closest_states = closest_states.reshape(-1, np.shape(full_data)[1])
                # check how many of the closest states are in the GP
                in_gp=self.in_gp[idx]

            size=np.minimum(closest_states.shape[0],10)
            # If we do not have enough closest states in the GP, add them to the GP
            if np.sum(in_gp)<=size:
                X_prev = self.gps[0].X
                update_gp=True
                extra_points=np.random.choice(np.arange(closest_states.shape[0]),size=size,replace=False)
                X_extra=closest_states[extra_points,:]
                Y_extra=closest_y[extra_points,:]

                X_new=np.vstack((X_prev,X_extra))


            # Check if applying other actions guarantee safety --> Pick safe actions from the closest state
            alternative_options = np.zeros([np.shape(closest_states)[0], np.shape(full_data)[1]])
            alternative_options[:, :-self.state_dim] = closest_states[:, :-self.state_dim]
            alternative_options[:, self.state_idx] = state



        # Same process as above but we only consider points in S
        else:
            diff = np.linalg.norm(self.S[:, self.state_idx] - state, axis=1)
            closest_states_idx = np.where(diff == diff.min())[0]
            closest_states = self.S[closest_states_idx, :]
            closest_states = closest_states.reshape(-1, np.shape(self.S)[1])

            # Check if applying other actions guarantee safety
            alternative_options = np.zeros([np.shape(closest_states)[0], np.shape(self.S)[1]])
            alternative_options[:, :-self.state_dim] = closest_states[:, :-self.state_dim]
            alternative_options[:, self.state_idx] = state

        # If we want to add new data points to our GP
        if update_gp:
            for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                if self.fmin[i]==-np.inf:
                    continue
                # Add new closest states to the GP
                y=gp.Y
                ynew=np.vstack((y,Y_extra[:,i].reshape(-1,1)))
                gp.set_XY(X_new,ynew)
                mean, var = gp.predict_noiseless(alternative_options)
                mean = mean.squeeze()
                std_dev = np.sqrt(var.squeeze())
                lower_bound = np.atleast_1d(mean - beta * std_dev)

                # If false, we satisfy constraints with a good tolerance
                at_boundary= np.all(lower_bound-self.fmin[i]-self.tol*std_dev-self.eta*scaling<0) # There is no action safe enough with a tolerance

                if at_boundary:
                    # Store state in the failed state list
                    self.Failed_state_list.append(state)
                    # Find best alternative safe action
                    action=self.find_constraint_max(alternative_options[:,:self.action_dim],state)
                    # Reset GP data
                    gp.set_XY(X_prev, y)
                    break
                
                gp.set_XY(X_prev, y)

        else:

            for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                # Add new closest states to the GP
                if self.fmin[i] == -np.inf:
                    continue
                mean, var = gp.predict_noiseless(alternative_options)
                mean = mean.squeeze()
                std_dev = np.sqrt(var.squeeze())
                lower_bound = np.atleast_1d(mean - beta * std_dev)

                # If false, we satisfy constraints with a good tolerance
                at_boundary = np.all(lower_bound - self.fmin[
                    i] - self.tol * std_dev - self.eta * scaling < 0)  # There is no action safe enough with a tolerance

                if at_boundary:
                    # Store state in the failed state list
                    self.Failed_state_list.append(state)
                    # Find best action
                    action = self.find_constraint_max(alternative_options[:, :self.action_dim], state)
                    break



        return at_boundary,Fail,action

    def add_boundary_points(self,x):
        """""
            Adds point x to the boundary list
            Parameters
            ----------
                x: ndarray
                    A state and action pair which is to be added to the boundary list
        """""
        self.Failed_experiment_list.append(x)


    def update_boundary_points(self):
        """""
           Checks if failed experiments (epxeriments where we hit the boundary) 
           would still fail after updating GPs and safe sets.
           If yes, these experiments are removed from the GP
        """""
        # Check if we have any failed experiments
        if self.Failed_experiment_list:
            beta=self.beta(self.t)

            # Indicates if we are using the full data (from GP and safe set) to evaluate safety
            if self.check_full_data:
                # Loop over all states at which we hit the boundary -> In reverse order, allows popping of list
                #full_data = np.vstack((self._x, self.S))
                full_data=self._x
                for i,state in reversed(list(enumerate(self.Failed_state_list))):

                    # Find the set of close states as done for the boundary condition function
                    diff = np.abs(self.S[:, self.state_idx] - state)
                    diff = (diff - self.optimal_velocities[self.state_idx]) <= 0
                    idx = np.sum(diff, axis=1) == self.state_dim
                    if np.sum(idx) > 0:
                        lower_bounds = self.lower_bound[idx, :]
                        slack = lower_bounds - self.fmin - self.eta * self.scaling * 1.2
                        at_boundary=bool(1-np.any(np.sum(slack>0,axis=1) == len(self.gps)))
                        if not at_boundary:
                            self.Failed_experiment_list.pop(i)
                            self.Failed_state_list.pop(i)
                            continue
                    # Find closest state in the full dataset
                    update_gp = False
                    diff = np.abs(full_data[:, self.state_idx] - state)
                    diff=(diff-self.optimal_velocities[self.state_idx])<=0
                    idx = np.sum(diff, axis=1) == self.state_dim
                    if np.sum(idx) == 0:
                        diff = np.linalg.norm(full_data[:, self.state_idx] - state, axis=1)
                        closest_states_idx = np.where(diff == diff.min())[0]
                        closest_states = full_data[closest_states_idx, :]
                        closest_y = self._y[closest_states_idx, :]
                        closest_states = closest_states.reshape(-1, np.shape(full_data)[1])
                        in_gp = self.in_gp[closest_states_idx]
                    else:
                        closest_states = full_data[idx, :]
                        closest_y = self._y[idx, :]
                        closest_states = closest_states.reshape(-1, np.shape(full_data)[1])
                        in_gp = self.in_gp[idx]

                    # Check if applying other actions guarantees safety
                    alternative_options = np.zeros([np.shape(closest_states)[0], np.shape(full_data)[1]])
                    alternative_options[:, :-self.state_dim] = closest_states[:, :-self.state_dim]
                    alternative_options[:, self.state_idx] = state

                    # Check if we have enough data points for the state in the GP
                    size = np.minimum(closest_states.shape[0], 10)
                    if np.sum(in_gp) <= size:
                        X_prev = self.gps[0].X
                        update_gp = True
                        # Y_prev=np.zeros([X_prev.shape[0],len(self.gps)])
                        extra_points = np.random.choice(np.arange(closest_states.shape[0]), size=size, replace=False)
                        X_extra = closest_states[extra_points, :]
                        Y_extra = closest_y[extra_points, :]

                        X_new = np.vstack((X_prev, X_extra))

                    # If we want to updat the GP to evaluate the boundary condition
                    if update_gp:
                        for j, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                            if self.fmin[j] == -np.inf:
                                continue
                            y = gp.Y
                            ynew = np.vstack((y, Y_extra[:, j].reshape(-1, 1)))
                            # update the GP
                            gp.set_XY(X_new, ynew)
                            mean, var = gp.predict_noiseless(alternative_options)
                            mean = mean.squeeze()
                            std_dev = np.sqrt(var.squeeze())
                            lower_bound = np.atleast_1d(mean - beta * std_dev)

                            # If false, we satisfy constraints with a good tolerance
                            at_boundary = np.all(lower_bound - self.fmin[j] - self.tol * std_dev - self.eta * scaling < 0)
                            # Reset the GP
                            gp.set_XY(X_prev, y)
                            # If true, we do not need to evaluate the other constraints
                            if at_boundary:
                                break
                        # If we do not hit the boundary now, remove these points from the list
                        if not at_boundary:
                            self.Failed_experiment_list.pop(i)
                            self.Failed_state_list.pop(i)

                    else:
                        for j, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                            if self.fmin[j] == -np.inf:
                                continue
                            mean, var = gp.predict_noiseless(alternative_options)
                            mean = mean.squeeze()
                            std_dev = np.sqrt(var.squeeze())
                            lower_bound = np.atleast_1d(mean - beta * std_dev)

                            # If false, we satisfy constraints with a good tolerance
                            at_boundary = np.all(lower_bound - self.fmin[j] - self.tol * std_dev - self.eta * scaling < 0)

                            # If true, we do not need to evaluate the other constraints
                            if at_boundary:
                                break
                        # If we do not hit the boundary now, remove these points from the list
                        if not at_boundary:
                            self.Failed_experiment_list.pop(i)
                            self.Failed_state_list.pop(i)


            # Same process as above but we only look safe states in S
            else:
                for i,state in reversed(list(enumerate(self.Failed_state_list))):

                    # Find the closest safe state
                    diff = np.abs(self.S[:, self.state_idx] - state)
                    diff=(diff-self.optimal_velocities[self.state_idx])<=0
                    idx = np.sum(diff, axis=1) == self.state_dim
                    if np.sum(idx) == 0:
                        diff = np.linalg.norm(self.S[:, self.state_idx] - state, axis=1)
                        closest_states_idx = np.where(diff == diff.min())[0]
                        closest_states = self.S[closest_states_idx, :]
                        closest_states = closest_states.reshape(-1, np.shape(self.S)[1])
                    else:
                        closest_states = self.S[idx, :]
                        closest_states = closest_states.reshape(-1, np.shape(self.S)[1])

                    # Check if applying other actions guarantee safety
                    alternative_options = np.zeros([np.shape(closest_states)[0], np.shape(self.S)[1]])
                    alternative_options[:, :-self.state_dim] = closest_states[:, :-self.state_dim]
                    alternative_options[:, self.state_idx] = state

                    for j, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
                        mean, var = gp.predict_noiseless(alternative_options)
                        mean = mean.squeeze()
                        std_dev = np.sqrt(var.squeeze())
                        lower_bound = np.atleast_1d(mean - beta * std_dev)

                        # If false, we satisfy constraints with a good tolerance
                        at_boundary = np.all(lower_bound - self.fmin[j] - self.tol * std_dev - self.eta * scaling < 0)

                        if at_boundary:
                            continue

                    if not at_boundary:
                        self.Failed_experiment_list.pop(i)
                        self.Failed_state_list.pop(i)

    def select_gp_subset(self,method=1):
        """""
                   
                   Used to reduce computational complexity of inference.
                   
                   Parameters
                   ----------
                   method: int
                        This parameter controls the method we use for picking our data points:
            
                            * 1 : Runs swarm optimization with 300 particles and selections self.boundary_ratio*N_reset
                                  unique points which have high covariance with the particles obtained from swarm 
                                  optimization. These are considered as boundary particles. Remaining particles are
                                  picked to be the  ones with low covariance with the particles obtained from swarm
                                  optimization. These are considered to be interior particles
                            * 2 : Evaluates exp(-5*ln^2)sigma_n for each point collected so far, picks self.boundary_ration*N_reset
                                  points with the largest values and the remaining points are chosen at random from the dataset
                                  
         """""
        # Find all points with IC other than the ones close to x_0
        X = self._x
        Y=self._y
        #idx_rest = [x for x in list(range(X.shape[0])) if x not in self.x_0_idx_gp_full_data]
        diff=np.abs(self._x[:,self.state_idx]-self.x_0)
        diff = (diff - self.optimal_velocities[self.state_idx]) > 0
        idx_rest = np.sum(diff, axis=1) >0
        idx_rest=np.arange(X.shape[0],dtype=int)[idx_rest]
        if np.sum(idx_rest)<self.N_reset:
            idx_rest = [x for x in list(range(X.shape[0])) if x not in self.x_0_idx_gp_full_data]
        # Check if non x_0 points are greater than N_reset
        if len(idx_rest)> self.N_reset:
            # Determine number of boundary points we want for our GP
            set_size = int(self.N_reset * self.boundary_ratio)
            # Calculate the remaining interior points
            interior_size = self.N_reset - set_size
            X = X[idx_rest, :]
            Y = Y[idx_rest, :]
            # Swarm Optimization Method
            if method==1:
                # Get expander points
                expander_points=self.get_boundary_particles(total_size=300)
                np.random.shuffle(expander_points)
                # np.random.shuffle(interior_particles)
                # # Determine covariance between the data and expander points,
                # # here we take the minimum of the covariance between the GPs
                covariance=self.compute_particle_distance(expander_points,X,full=True)
                # # Find unique points with the highest covariance -> Boundary points
                idx_boundary=np.argmax(covariance,axis=1)
                idx_boundary=np.unique(idx_boundary)
                if idx_boundary.shape[0] >= set_size:
                     idx_boundary = np.random.choice(idx_boundary,set_size,replace=False)
                     idx_boundary=idx_boundary.reshape(-1,1)
                else:
                    idx_boundary = idx_boundary.reshape(-1, 1)
                    # # Get covariance ranking matrix
                    covariance_rankings=np.argsort(np.argsort(covariance))
                    n_points=X.shape[0]
                    # idx_full=np.unique(idx_full).reshape(-1,1)
                    # # Loop over all candidate to select subset
                    for i in range(1,n_points):
                        # If enough data points for the boundary and interior have been selected, exit the loop
                        if idx_boundary.shape[0]>= set_size:
                            break
                        # If more boundary points are required, pick those which have a high covariance with the expanders
                        # High rank
                        rows,idx=np.where(covariance_rankings==n_points -i-1)
                        idx=np.unique(idx)
                        # Add new points to the boundary idx, make sure they are all unique
                        idx_boundary=np.vstack((idx_boundary,idx.reshape(-1,1)))
                        idx_boundary=np.unique(idx_boundary)
                        idx_boundary=np.asarray(idx_boundary)
                        idx_boundary=idx_boundary.reshape(-1,1)






                # Randomly pick the remaining points, here considered to be the interior points
                idx_interior=[x for x in np.arange(X.shape[0]) if x not in idx_boundary]
                idx_interior=np.asarray(idx_interior)
                X_exterior=X[idx_boundary.squeeze(),:]
                X_interior=X[idx_interior,:]

                # Pick points based on a probability proportional to -covariance with boundary points
                covariance_interior=self.compute_particle_distance(X_interior,np.vstack((X_exterior,self._x[self.x_0_idx_gp_full_data,:])))
                #covariance_interior=np.max(covariance_interior,axis=1)
                p=np.exp(-5*covariance_interior)
                p/=np.sum(p)


                idx_interior=np.random.choice(idx_interior,interior_size,p=p,replace=False)

                # import matplotlib.pyplot as plt
                # fig_plot = plt.figure(figsize=(14, 14))
                # left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                # ax_plot = fig_plot.add_axes([left, bottom, width, height])
                # ax_plot.scatter(self._x[:, 0], self._x[:, 1], c="Darkred", label="Remaining points")
                # ax_plot.scatter(X[idx_boundary, 0], X[idx_boundary, 1], c="blue", label="Boundary Points")
                # ax_plot.scatter(X[idx_interior, 0], X[idx_interior, 1], c="green", label="Interior Points")
                # ax_plot.set_title('All data points and data points in GP')
                # #
                # ax_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                #                 fancybox=True, shadow=True, ncol=5, title="Data selection scheme")
                # ax_plot.set_xlabel('a')
                # ax_plot.set_ylabel('x')
                # ax_plot.set_xlim([-6.5, 6.5])
                # ax_plot.set_ylim([-0.8, 0.8])
                # name = "Data points in GP after" + ".png"
                # fig_plot.savefig(name, dpi=300)
                # Determine the full data we are going to use for the GP
                idx_full=np.vstack((idx_boundary,idx_interior.reshape(-1,1)))
                X_full=X[idx_full,:]
                X_full=X_full.squeeze()
                X_full = np.vstack((self._x[self.x_0_idx_gp_full_data,:],X_full))

                Y_full=Y[idx_full,:]
                Y_full = Y_full.squeeze()
                Y_full = np.vstack((self._y[self.x_0_idx_gp_full_data, :],Y_full))


                # Update the in_gp vector -> True if the data point is in GP
                self.in_gp[:]=False
                self.in_gp[self.x_0_idx_gp_full_data]=True
                remainder=self.in_gp[idx_rest]
                remainder[idx_full]=True
                self.in_gp[idx_rest]=remainder


            # apply scheme 2-> Define objective function and pick best points based on
            # the objective
            elif method==2:
                # Method:
                # Evaluate objective function for each data point
                beta=self.beta(self.t)
                interest_function = (len(self.gps) *
                                     np.ones(X.shape[0], dtype=np.float))
                values=np.zeros(X.shape[0])
                slack_min=np.ones(X.shape[0])*np.inf
                for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):

                    if self.fmin[i] == -np.inf:
                        continue
                    mean, var = gp.predict_noiseless(X)
                    mean = mean.squeeze()
                    std_dev = np.sqrt(var.squeeze())
                    lower_bound = mean - beta*std_dev
                    slack=np.atleast_1d(lower_bound - self.fmin[i])
                    slack /=scaling
                    values=np.maximum(values,std_dev/scaling)
                    slack_min=np.minimum(slack,slack_min)


                # Determine the interest function
                interest_function*=norm.pdf(slack_min,scale=0.2)

                # function to sort the array based on maximum values
                def sort_generator(array):
                    """Return the sorted array, largest element first."""
                    return array.argsort()[::-1]
                # Sort the objective by interest_function*values
                score=interest_function*values
                idx_sorted = sort_generator(score)
                # Pick the boundary points as the ones with the highest values
                expander_size=int(set_size*0.75)
                idx_boundary = idx_sorted[:set_size]
                idx_sorted2=sort_generator(interest_function) #Points with low lowerbounds only
                idx_boundary2=idx_sorted2[:set_size-expander_size]
                idx_boundary=np.vstack((idx_boundary.reshape(-1,1),idx_boundary2.reshape(-1,1)))
                # Sample interior points at random from the remaining ones

                idx_interior = [x for x in list(range(X.shape[0])) if x not in idx_boundary]
                idx_interior=np.random.choice(idx_interior,size=interior_size)
                idx_full = np.vstack((idx_boundary.reshape(-1,1), idx_interior.reshape(-1,1)))



                # Define full data matrix for the GP
                X_full = X[idx_full, :]
                X_full = X_full.squeeze()
                X_full = np.vstack((self._x[self.x_0_idx_gp_full_data, :], X_full))

                Y_full = Y[idx_full, :]
                Y_full = Y_full.squeeze()
                Y_full = np.vstack((self._y[self.x_0_idx_gp_full_data, :], Y_full))

            # Update gp idx
            self.x_0_idx_gp = np.arange(start=0, stop=self._x[self.x_0_idx_gp_full_data, :].shape[0], step=1)

            # set GP points
            for i, gp in enumerate(self.gps):
                gp.set_XY(X_full,Y_full[:,i].reshape(-1,1))




    def get_boundary_particles(self,total_size,Fast=False,iters='auto'):
        """""
           Uses swarm optimization to obtain points from G_n and partial S_n.
           
           Parameters
           -----------
           total_size: int
                total number of particles
            
            Returns
            -------
            best_particles: ndarray
                Set of particles which are the local minimas of the boundary swarms
                for each safe region
            """""

        def sort_generator(array):
            """Return the sorted array, largest element first."""
            return array.argsort()[::-1]

        # Find all sets in the safe set
        sets=np.unique(self.set_number)
        # Partition total size equally among all sets
        total_sets=len(sets)
        size_per_set=int(total_size/total_sets)

        best_particles=[]

        # Define fraction size of expander particles per set
        size_expander=int(0.75*size_per_set)

        # Update bounds for the optimization -> Important such that all points in the GP are considered
        bounds=self.bounds.copy()
        # No clipping of particles for state dim
        bounds=np.asarray(bounds)
        max_states = np.maximum(np.max(np.abs(self._x[:, self.state_idx]), axis=0)* 1.5, bounds[self.state_idx, 1])
        bounds[self.state_idx, 0] = -max_states
        bounds[self.state_idx, 1] = max_states
        bounds = bounds.tolist()

        # Define swarms for the expander and boundary
        swarm_expanders = SwarmOptimization(size_expander,self.optimal_velocities,
                                                        partial(self._boundary_fitness_function,
                                                                'expander'),
                                                        bounds=bounds)


        swarm_boundary_states=SwarmOptimization(size_per_set-size_expander,self.optimal_velocities,
                                                partial(self._boundary_fitness_function,
                                                        'boundary'),
                                                        bounds=bounds)
        # Avoid driving points together by encouraging velocities towards global optimum
        swarm_expanders.c2 = 0
        swarm_boundary_states.c2=0

        # Determine for each point in the safe set their slack
        # This is used to then sample these particles
        lower_bound=self.lower_bound-self.fmin
        slack_min=np.min(lower_bound,axis=1)
        slack_min= np.atleast_1d(slack_min)

        #dist=self.compute_particle_distance(self.S,self.S[self.x_0_idx,:])

        # Calculate unnormalized probability measure for sampling
        p=np.exp(-5*slack_min**2)
        if iters=='auto':
            iterations=int(self.max_iters/len(sets))
        else:
            iterations=self.max_iters
        #n_points = X.shape[0]
        for i,number in enumerate(sets):
            # Perform swarm optimization and pick best particles

            # Find points corresponding to the appropriate set number
            indexes=np.where(self.set_number == number)[0]
            # Determine sampling probability
            p_points = p[indexes]
            if np.sum(p_points) == 0:
                p_points= np.ones(len(p_points)) * 1 / len(p_points)
            else:
                p_points /= np.sum(p_points)


            # Randomly sample points to initialize particles for the swarm
            random_id = np.random.choice(indexes, size=size_per_set,p=p_points)
            particles = self.S[random_id, :]


            # Perturb particles for diversity
            if self.perturb_particles:
                u = np.random.uniform(-1, 1, size=particles.shape)
                particles = particles + u * self.optimal_velocities



            # Run and initialize swarms
            swarm_expanders.init_swarm(particles[:size_expander])
            swarm_expanders.run_swarm(iterations)

            idx_sorted=sort_generator(swarm_expanders.best_values)

            best_expander=swarm_expanders.best_positions[idx_sorted]


            if self.use_convexhull:
                # Define the convex hull and read out the vertices
                hull=ConvexHull(best_expander)
                expanders=best_expander[hull.vertices,:]


                # Find potential points which we could add (cov <0,95)
                cov=self.compute_particle_distance(best_expander,expanders)
                idx=np.where(cov<0.95)[0]

                # If we want to have a fast approach, add all the potential points. This is fast but points are not
                # as wide spread. Else add point 1 by 1
                if Fast:
                    best_expander = np.vstack((expanders, best_expander[idx, :]))
                else:
                    # Loop over all potential points and if the covariance is < 0.95, add it to the selected points
                    selected_points=expanders
                    for index in idx:
                        cov=self.compute_particle_distance(best_expander[index,:].reshape(1,-1),selected_points)

                        if cov<0.95:
                            selected_points = np.vstack((selected_points, best_expander[index, :]))

                    best_expander=selected_points
                
            # Follow the same procedure for the boundary particles
            swarm_boundary_states.init_swarm(particles[size_expander:])
            swarm_boundary_states.run_swarm(iterations)

            idx_sorted = sort_generator(swarm_boundary_states.best_values)
            best_boundary = swarm_boundary_states.best_positions[idx_sorted]

            if self.use_convexhull:
                hull = ConvexHull(best_boundary)
                boundary= best_boundary[hull.vertices, :]


                cov = self.compute_particle_distance(best_boundary, boundary)
                idx = np.where(cov < 0.95)[0]

                if Fast:
                    best_boundary = np.vstack((boundary, best_boundary[idx, :]))
                else:
                    selected_points = boundary
                    for index in idx:
                        cov = self.compute_particle_distance(best_boundary[index, :].reshape(1, -1), selected_points)

                        if cov < 0.95:
                            selected_points = np.vstack((selected_points, best_boundary[index, :]))

                    #
                    best_boundary=selected_points




            # Store bestparticles for each swarm
            best_points=np.vstack((best_expander,best_boundary))
            best_points=unique(best_points)
            best_particles.extend(best_points)

        return np.asarray(best_particles)




    def _boundary_fitness_function(self,swarm_type,particles):
        """
        Swarm optimization objective, used for the swarm which approximates partial S_n

        Parameters
        ----------
        particles: ndarray
            Particles for which the objective should be evaluated

        Returns
        -------
        values: ndarray
            Evaluated objective for each particles

        global_safe: ndarray
            Boolean array indicating if the particle
            is safe
        """
        beta = self.beta(self.t)
        interest_function = (len(self.gps) *
                             np.ones(particles.shape[0], dtype=np.float))

        total_penalty=np.zeros(particles.shape[0],dtype=np.float)
        slack_min=np.ones(len(particles))*np.inf
        global_safe = np.ones(particles.shape[0], dtype=np.bool)

        #x_0_particle=np.zeros([1,particles.shape[1]],dtype=np.float)
        #x_0_particle[:,self.state_idx]=self.x_0

        #state_vec=particles.copy()
        #state_vec[:,:self.action_dim]=0

        #dist=self.compute_particle_distance(state_vec,x_0_particle)
        #interest_function*=np.exp(-0.1*dist)

        # Loop over all GPs associated with the constraints
        max_var=np.zeros(particles.shape[0],dtype=np.float)
        for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):

            if self.fmin[i] == -np.inf:
                continue
            # Determine the slacks
            mean, var = gp.predict_noiseless(particles)
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())
            lower_bound = mean - beta * std_dev

            slack = np.atleast_1d(lower_bound - self.fmin[i])
            if swarm_type == "expander":
                max_var=np.maximum(max_var,std_dev/scaling)
            global_safe &= slack >= 0
            slack /= scaling
            total_penalty += self._compute_penalty(slack)
            # Take the minimum slack
            slack_min=np.minimum(slack_min,slack)

        # Calculate the objective value
        interest_function *= norm.pdf(slack_min, scale=0.2)
        #interior_particles = self.S[self.inner_particles, :]
        #covariance = self.compute_particle_distance(particles, interior_particles)
        #interest_function *= np.exp(-5 * covariance)
        if swarm_type=="expander":
            values=interest_function*(total_penalty+max_var)
        elif swarm_type=="boundary":
            values = interest_function * (total_penalty + 1)

        return values,global_safe



    def find_constraint_max(self,a_init,x,iters=None):
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
        iterations=iters
        if iters is None:
            iterations=self.max_iters

        # Define the objective function for the swarm
        constraint_objective=lambda a: self.constraint_objective(a,x=x)
        bounds = np.asarray(self.bounds)
        # Define the swarm
        swarm_maximizer = SwarmOptimization(self.swarm_size, self.S1_velocities,
                                                  constraint_objective,
                                                  bounds=bounds[:self.action_dim].tolist())

        # Pick particles for swarm initialization
        random_id=np.random.randint(a_init.shape[0], size=self.swarm_size)
        a=a_init[random_id,:]
        # perturb particles if desired
        if self.perturb_particles:
            u = np.random.uniform(-1, 1, size=a.shape)
            a = a + u * self.S1_velocities
            action_bounds = np.asarray(self.bounds[:-self.state_dim])
            action_bounds = action_bounds.T  # Lower upper bound are stored in each column
            a = np.clip(a, action_bounds[0], action_bounds[1])

        # Run swarm and find the global best
        swarm_maximizer.init_swarm(a)
        swarm_maximizer.run_swarm(iterations)
        global_best = swarm_maximizer.global_best[None, :]
        return global_best





    def constraint_objective(self,a,x):

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
        lower_bound=np.ones(a.shape[0])*np.inf
        beta = self.beta(self.t)
        global_safe=np.ones(a.shape[0],dtype=bool)
        #Loop over each GP to find the minimum slack
        for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
            if self.fmin[i]==-np.inf:
                continue

            mean, var = gp.predict_noiseless(particles)
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())
            lb= mean - beta * std_dev
            slack = np.atleast_1d(lb- self.fmin[i])
            lower_bound=np.minimum(lower_bound,slack/scaling)

        return lower_bound,global_safe
            



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
    tol: double
        tolerance used for boundary condition
    eta: double
        Additional tolerance element used for the boundary conditon
    eps: double
        Maximum precision for convergence
    max_data_size: int
        Maximum number of data points of states other than x_0 that
        the full GP can contain
    reset_size: int
        Number of data points we choose for our subset (only considering states
        other than x0).



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


        self.gp_full=gp_full

        # Read state information
        self.state_dim = np.shape(x_0)[0]
        self.state_idx = list(range(self.gp_full[0].input_dim - self.state_dim, self.gp_full[0].input_dim))
        self.x_0 = x_0.T


        self.action_dim=self.gp.input_dim
        assert self.action_dim+self.state_dim == self.gp_full[0].input_dim, 'state dim + action dim != Full GP input dim'
        S3_velocities = self.optimal_velocities.copy()
        S3_velocities = S3_velocities * 10
        self.swarms['S3']=SwarmOptimization(swarm_size,S3_velocities,partial(self._compute_particle_fitness,'S3'),bounds=bounds)
        self.criterion = 'init'
        # Boolean used to indicate if we should perturb particles for further exploration
        self.perturb_particles = True
        x_0_idx=np.asarray(range(len(self.S)))
        self.x_0_idx_gp_full = x_0_idx.copy()
        # All combinations of x_0 in all the recorded data
        self.x_0_idx_full_data = x_0_idx.copy()
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
        self.set_number = np.zeros(self.S.shape[0], dtype=int)
        self.current_set_number = 0
        self.eps = eps
        self.data_size_max = max_data_size
        self.N_reset = reset_size
        # Lower bounds for all the data points collected so far
        self.lower_bound = np.ones([self._x.shape[0], len(self.gps)]) * -np.inf
        # Interior points: Points that fulfill the boundary condition
        self.interior_points=np.zeros(self.lower_bound.shape[0])
        self.marginal_points = np.zeros(self.lower_bound.shape[0])
        self.Update_data_lower_bounds()

        # Boundary thresshold, used to evaluate boundary condition
        self.boundary_thresshold_u = boundary_thresshold_u
        self.boundary_thresshold_l = boundary_thresshold_l
        # Indicates if safe action should be returned if the boundary is hit.
        self.return_safe_action=True
        self.L_states=L_states
        self.state_velocities_u=self.optimal_state_velocity(self.boundary_thresshold_u)
        self.state_squaraed_dist_u=np.sum(np.square(self.state_velocities_u/self.L_states))

        if self.use_marginal_set:
            self.state_velocities_l = self.optimal_state_velocity(self.boundary_thresshold_l)
            self.state_squaraed_dist_l = np.sum(np.square(self.state_velocities_l / self.L_states))
        self.encourage_jumps = True
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
        """Optimize the velocities of the particles.

        Note that this only works well for stationary kernels and constant mean
        functions. Otherwise the velocity depends on the position!

        Returns
        -------
        velocities: ndarray
            The estimated optimal velocities in each direction.
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
                interest_function = (len(self.gps) *
                                     np.ones(np.shape(values), dtype=np.float))
                X_data = np.vstack((self.gp.X, self.S))
                covariance = self.compute_particle_distance(particles, X_data)
                if self.Failed_experiment_list:

                    failed_experiments = np.asarray(self.Failed_experiment_list.copy())
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
                bound = np.array(self.bounds)
                action = np.random.uniform(low=bound[:, 0], high=bound[:, 1],
                                           size=[self.swarm_size, self.action_dim])
                particles = action.reshape(-1, self.action_dim)
                current_set = self.current_set_number
            else:
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

        self.Update_data_lower_bounds()
        # compute estimate of the lower bound
        self.greedy, self.best_lower_bound = self.get_new_query_point('greedy')

        if self.gp_full[0].X.shape[0]-self.gp_full[0].X[self.x_0_idx_gp_full,:].shape[0]>=self.data_size_max:
            self.select_gp_subset()

        if self.encourage_jumps:
            # S3 has not be successful, reduce S1 steps.
            max_S1_step=np.minimum(20,self.max_S1_steps)
            if len(np.where(self.set_number == self.current_set_number)[0])>0:
                max_S1_step=self.max_S1_steps
        # Check if we have exceeded maximum number of steps for S1, if yes go to S3
        if self.s1_steps<max_S1_step:
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
            if max(std_exp,std_maxi)>self.eps:

                logging.info("The best maximizer has std. dev. %f" % std_maxi)
                logging.info("The best expander has std. dev. %f" % std_exp)
                logging.info("The greedy estimate of lower bound has value %f" %
                             self.best_lower_bound)

                if std_maxi >= std_exp:
                    return x_maxi.squeeze()
                else:
                    return x_exp.squeeze()


        self.criterion = "S3"
        # If self.s3_steps=0 -> We are running S3 for the first time after exploring the previous set,
        # Update set number
        if self.s3_steps==0:
            self.current_set_number+=1

        self.s3_steps += 1
        # Check if we have exceeded maximum number of S3 steps, if yes reset all step counters, if no sample from S3 swarm
        self.switch= self.s3_steps>=self.max_S3_steps
        if self.switch:
            self.s1_steps=0
            self.s3_steps=0
            self.switch = False



        x_exp, std_exp = self.get_new_query_point('S3')
        std_exp /= self.scaling
        std_exp = np.max(std_exp)
        logging.info("The best S3 point has std. dev. %f" % std_exp)
        return x_exp.squeeze()



    def add_new_data_point(self,x,y):
        '''
        Adds new data points to full gp and local gp (only for parameter).
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
        # If state corresponds to x0
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
        initial_size_full_data = self._x.shape[0]
        initial_size_gp_full = self.gp_full[0].X.shape[0]
        is_x0=np.sum(x[:,self.state_idx]==self.x_0.squeeze(),axis=1)==self.state_dim
        a = x[:, :self.action_dim]
        if self.criterion=="S3" and is_x0[0]:
            self.criterion == "init"
            self.S = np.vstack((self.S, a[0,:]))
            self.s3_steps=0
            self.s1_steps=0
            self.set_number = np.vstack((self.set_number, self.current_set_number))

        a=np.atleast_2d(a)
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        a=a[is_x0,:]
        y_x0=y[is_x0,:]

        self._x = np.concatenate((self._x, x), axis=0)
        self._y = np.concatenate((self._y, y), axis=0)
        add_to_data = np.ones(x.shape[0], dtype=bool)
        self.in_gp_full = np.append(self.in_gp_full, add_to_data)
        for i, gp in enumerate(self.gps):
            not_nan = ~np.isnan(y[:, i])
            if np.any(not_nan):
                self.gp_full[i].set_XY(np.vstack([self.gp_full[i].X, x[not_nan, :]]),
                          np.vstack([self.gp_full[i].Y, y[not_nan, i].reshape(-1, 1)]))

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
        beta=self.beta(self.t)
        difference=self._x.shape[0]-self.lower_bound.shape[0]
        self.lower_bound=np.vstack((self.lower_bound,np.ones([difference,len(self.gps)])*-np.inf))
        # Evaluate lower bound and update constrained set lower bounds if increasing.
        for i, gp in enumerate(self.gp_full):
            if self.fmin[i]==-np.inf:
                continue
            mean, var = gp.predict_noiseless(self._x)
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())
            lower_bound = mean - beta * std_dev
            self.lower_bound[:,i]=np.maximum(self.lower_bound[:, i], lower_bound)

        # Determine interior points: Points at which we will not hit the boundary
        constraint_idx=np.where(self.fmin != -np.inf)
        constraint_idx=np.asarray(constraint_idx).reshape(-1,1)
        slack=self.lower_bound[:,constraint_idx]-self.fmin[constraint_idx]
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
        # Check voariance between interior states and current state
        diff = interior_states[:, self.state_idx] - state
        diff = diff / self.L_states
        squared_dist = np.sum(np.square(diff), axis=1)
        if self.use_marginal_set:
            marginal_idx=self.marginal_points[self.interior_points]
            safe = squared_dist <= self.state_squaraed_dist_l
            safe[marginal_idx]&=squared_dist[marginal_idx]<=self.state_squaraed_dist_u
        else:
            safe=squared_dist<=self.state_squaraed_dist_u
        #diff = (diff - self.state_velocities) <= 0
        #safe = np.sum(diff, axis=1) == self.state_dim
        #alternative=interior_states.copy()
        #alternative[:,self.state_idx]=state
        #covariance=self.gp_full[0].kern.K(alternative,interior_states)
        #covariance /= self.scaling[0] ** 2
        #covariance = np.max(covariance, axis=1)
        a_safe=None
        # If covariance > some thresshold, then we are not at the boundary
        #if np.any(covariance >=self.boundary_thresshold):
        if np.sum(safe) > 0:
            at_boundary=False
        else:
            at_boundary=True
        # If at the boundary and an alternative safe action is asked, return the best action seen so far
        if at_boundary:
            self.Failed_state_list.append(state)
            if self.return_safe_action:
                if not self.fast_safe_action:
                    idx_action = np.where(squared_dist == squared_dist.min())[0]
                    a_init=interior_states[idx_action,:self.action_dim].copy()
                    a_safe = self.find_constraint_max(a_init, state)
                else:
                    idx_action=np.where(squared_dist==squared_dist.min())[0]
                    if len(idx_action)>1:
                        idx_max=np.argmax(self.lower_bound[idx_action, 0])
                        idx_action=idx_action[idx_max]
                    #a_safe,f = self.get_maximum()
                    a_safe=interior_states[idx_action,:self.action_dim]
                #a_safe, f = self.get_maximum()

        return at_boundary,a_safe


    def add_boundary_points(self,x):
        self.Failed_experiment_list.append(x.copy())

    def update_boundary_points(self):
        """""
           Checks if failed experiments (epxeriments where we hit the boundary) 
           would still fail after updating GPs and safe sets.
           If yes, these experiments are removed from the GP
        """""
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
                    #alternative = interior_states.copy()
                    #alternative[:,self.state_idx] = state
                    #covariance = self.gp_full[0].kern.K(alternative, interior_states)
                    #covariance /= self.scaling[0] ** 2
                    #covariance = np.max(covariance, axis=1)
                    #if np.any(covariance >= self.boundary_thresshold):
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
                    #alternative = interior_states.copy()
                    #alternative[:,self.state_idx] = state
                    #covariance = self.gp_full[0].kern.K(alternative, interior_states)
                    #covariance /= self.scaling[0] ** 2
                    #covariance = np.max(covariance, axis=1)
                    #if np.any(covariance >= self.boundary_thresshold):
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
        random_id = np.random.choice(idx_rest,size=self.N_reset, p=prob)
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
































