import numpy as np
import GPy
from safeopt import linearly_spaced_combinations
from safeopt import GoSafe


x=np.array([[1,-1,1],[1,1,1]])
y=np.array([[5],[3]])
bounds=[[-1,1],[-1,1],[-1,1]]
parameter_set = linearly_spaced_combinations(bounds, num_samples=100)
gp = GPy.models.GPRegression(x,y, noise_var=0.01**2)
gp2=GPy.models.GPRegression(x,0.5*y, noise_var=0.01**2)
opt = GoSafe([gp,gp2], parameter_set, fmin=[-np.inf,0],x_0=np.array([-1,1]))
opt.update_confidence_intervals()
opt.compute_safe_set()
#opt.compute_safe_states()
check=opt.at_boundary(state=np.array([0.6,0.4]))
x=2