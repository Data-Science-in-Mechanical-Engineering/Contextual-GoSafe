import numpy as np
import matplotlib.pyplot as plt
import GPy
from safeopt import linearly_spaced_combinations
from safeopt import Contextual_GoSafe
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import time
'''
own example
'''
plt.rcParams.update({'font.size': 16})
import random

#warnings.filterwarnings("ignore")
random.seed(10)
np.random.seed(10)






class mod_sys:
	"""
	class used to define 1D nonlinear system

	"""

	def __init__(self,k_init=0,x_des=0,high=0.8):
		self.a = 1.01
		self.b = -0.2
		self.k = k_init
		self.state = np.array([[0]])
		self.x_des = x_des
		high_obs = np.array([high])
		self.state_limits = list(zip(-high_obs,high_obs))

	def reset(self,x_0=None):
		if x_0 is None:
			self.state = np.array([[0]])
		else:
			self.state=np.array([[x_0]])


	def step(self):
		v = np.random.normal(0, 1e-4)
		w = np.random.normal(0, 1e-4)
		obs = self.state + w
		action = np.abs(self.k * (obs - self.x_des))
		self.state = self.a * np.sqrt(np.abs(self.state)) + self.b * np.sqrt(action) + v


def generate_heat_map(n_points):
	"""
	Performs grid search to generate a map for the objective and constraint function
	with respect to the parameters and initial states.
	"""
	overall_points=int(3*n_points/5)
	a1 = np.linspace(-6,6,overall_points)
	a2=np.linspace(-1,1,n_points-overall_points)
	a=np.hstack((a1,a2))
	a[::-1].sort()
	x_0 = np.linspace(-0.3,0.3,n_points)
	sys=mod_sys()
	n_steps=1000
	F=np.zeros([n_points,n_points])
	G=np.ones([n_points,n_points])*np.inf
	for k in range(n_points):
		for s in range(n_points):
			sys.reset(x_0[s])
			sys.k = a[k]
			F[k,s]=-sys.state**2
			G[k,s]=min(G[k,s],F[k,s])
			for j in range(n_steps):
				sys.step()
				cost=sys.state**2
				F[k,s]=F[k,s]-cost[0]
				G[k,s]=min(G[k,s],-cost[0])

	Data=np.zeros([n_points,2])
	Data[:,0]=a
	Data[:,1]=x_0

	return Data,F,G


def plot_function(n_points=101):
	"""
	Plots the objetive function and safe set.
	"""
	a = np.linspace(-6, 6, n_points)
	sys = mod_sys()
	n_steps = 1000
	f=np.zeros(n_points)
	g=np.ones(n_points)*np.inf
	for k in range(n_points):
		sys.reset()
		sys.k=a[k]
		f[k]=-sys.state**2
		g[k]=min(g[k],f[k])
		for j in range(n_steps):
			sys.step()
			cost = sys.state ** 2
			f[k] = f[k] - cost[0]
			g[k] = min(g[k], -cost[0])

	fig_g = plt.figure(figsize=(14, 14))
	left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
	ax = fig_g.add_axes([left, bottom, width, height])
	ax.plot(a, g, color="black", label="g")
	ax.axhline(y=-0.81, color='r', linestyle='-')
	ax.set_title('Constraint function')
	ax.set_xlabel('a')
	ax.set_ylabel('g')
	ax.set_xlim([-6.5, 6.5])
	name = "g.png"
	fig_g.savefig(name, dpi=300)

	fig_f = plt.figure(figsize=(14, 14))
	left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
	ax = fig_f.add_axes([left, bottom, width, height])
	ax.plot(a, f, color="black", label="g")
	ax.axhline(y=-800, color='r', linestyle='-')
	ax.set_title('objective function')
	ax.set_xlabel('a')
	ax.set_ylabel('f')
	ax.set_xlim([-6.5, 6.5])
	name = "f.png"
	fig_f.savefig(name, dpi=300)

class Optimizer(object):
	"""
	Defines the optimizer
	"""
	def __init__(self,initial_k=1,high=0.9,f_min=-np.inf,num_it=1000,lengthscale=0.5,ARD=True,variance=10000,eta=0.5):

		self.Fail = False
		self.at_boundary = False
		self.sys=mod_sys(high=high)
		self.high=high
		self.f_min=f_min
		self.cost_bound=min(-f_min,num_it)
		self.sys.k=initial_k
		self.num_it=num_it
		self.rollout_values=[0,0]
		self.rollout_limit = 10
		self.rollout_data = []
		# Define bounds for parameters
		bounds = [[-6, 5]]
		self.horizon_cost = np.zeros(self.num_it + 1)
        # Simulate to gather initial safe policy
		self.simulate()
		y1 = np.array([[self.rollout_values[0]]])
		y1 = y1.reshape(-1, 1)
		y2 = np.array([[self.rollout_values[1]]])
		y2 = y2.reshape(-1, 1)
		L  = [lengthscale,0.2]
        # GPs.
		a=np.asarray([[self.sys.k]])
		x = np.array([[self.sys.k, 0]])
		KERNEL_f = GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD,variance=variance)
		gp_full1 = GPy.models.GPRegression(x, y1, noise_var=0.01**2, kernel=KERNEL_f) #noise_var=0.01**2
		KERNEL_g=GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD,variance=1)
		gp_full2 = GPy.models.GPRegression(x, y2, noise_var=0.01 ** 2, kernel=KERNEL_g)

		KERNEL_f = GPy.kern.sde_Matern32(input_dim=a.shape[1], lengthscale=lengthscale, ARD=ARD, variance=variance)
		gp1 = GPy.models.GPRegression(a, y1, noise_var=0.01 ** 2, kernel=KERNEL_f)
		KERNEL_g = GPy.kern.sde_Matern32(input_dim=a.shape[1], lengthscale=lengthscale * 2, ARD=ARD, variance=1)
		gp2 = GPy.models.GPRegression(a, y2, noise_var=0.01 ** 2, kernel=KERNEL_g)



		# Set up optimizer
		L_states=L[1]
		self.opt=Contextual_GoSafe(gp=[gp1, gp2],gp_full=[gp_full1,gp_full2],L_states=L_states,bounds=bounds,fmin=[f_min, -high**2],x_0=np.array([[0]]),eta_L=eta,max_S1_steps=30,max_S3_steps=10,eps=0.1,max_data_size=100,reset_size=20) #worked for maxS2_steps=100
		params = np.linspace(2, 3, 2)
		self.initialize_gps(params)
		self.time_recorded=[]


	def reset(self,x_0=None):
	    """
	    Reset system to iniitial state
	    """
	    self.sys.reset(x_0)
	    self.Fail=False
	    self.at_boundary=False
	    self.rollout_values = [0, 0]
	    self.rollout_data=[]
	    self.horizon_cost = np.zeros(self.num_it + 1)

	def initialize_gps(self,params):
        
		for k in params:
			self.sys.k=k
			self.simulate()
			x = np.array([[self.sys.k, 0]])
			y = np.array([[self.rollout_values[0]], [self.rollout_values[1]]])
			y = y.squeeze()
			self.opt.add_new_data_point(x, y)

	def simulate(self,opt=None,x_0=None):
		"""
		Simulate system
		"""
		self.reset(x_0)
		f = -self.sys.state[0] ** 2
		g = f
		self.horizon_cost[0]=f
		for i in range(self.num_it):
			if i<self.rollout_limit:
				x = np.array([[self.sys.k, self.sys.state[0][0]]])
				self.rollout_data.append(x)

			cost, constraint = self.step(opt)
			if self.Fail:
				self.rollout_values = [f, g]
				print("Failed",end=" ")
				break
			f =f - cost[0]
			self.horizon_cost[i+1]=-cost[0]
			g = min(g, constraint[0])

		if not self.Fail:
			self.rollout_values=[f,g]
			print("function values",f,g,end=" ")


			
			
			
	def optimize(self):
		"""
		Perform 1 full optimization step
		"""
		self.opt.update_boundary_points()
		start_time = time.time()
		a = self.opt.optimize()
		self.time_recorded.append(time.time()-start_time)
		print(self.opt.criterion,a,end=" ")
		self.sys.k = a
		self.simulate(opt=self.opt)
		x = np.array([[a.squeeze(), self.opt.x_0.squeeze()]])
		y = np.array([[self.rollout_values[0]], [self.rollout_values[1]]])
		y = y.squeeze()

		if self.rollout_values[0]<=self.f_min or self.rollout_values[1]<=-self.high**2:
			print("hit constraint", end= " ")
		if not self.at_boundary:
			self.add_data(y)
			#self.opt.add_new_data_point(x,y)
		else:
			self.opt.add_boundary_points(a.reshape(1,-1))

		df2 = pd.DataFrame([[a, self.opt.x_0,self.rollout_values[0][0],self.rollout_values[1][0],self.opt.criterion,self.at_boundary,self.Fail]],
						   columns=['a',"x","f","g","criteria","boundary","Fail"])
		self.rollout_data = []
		return df2


	def step(self,opt=None):

		if opt is not None and not self.at_boundary:
			self.at_boundary, self.Fail, self.sys.k = opt.check_rollout(state=np.asarray([self.sys.state]).reshape(1,-1), action=self.sys.k)

			if self.Fail:
				return None,None

			elif self.at_boundary:
				print("Changed action to",self.sys.k,end=" ")

		self.sys.step()

		cost = self.sys.state ** 2
		constraint = -cost

		return cost, constraint

	def add_data(self,y):
		"""
		Add points to GP
		"""
		for i,x in enumerate(self.rollout_data):
			f_value=np.sum(self.horizon_cost[i:])
			y[0]=f_value
			self.opt.add_new_data_point(x, y)





def Optimize(num_experiments=91):
	"""
	Performs full optimization for num_experiments iterations
	"""

	plot = True
	if plot:
		log_dir = "./contextual"
		os.makedirs(log_dir, exist_ok=True)
		os.chdir(log_dir)
		Data, F, G = generate_heat_map(n_points=15)
		# # #np.savetxt('Function_map_mod_sys.csv', Data, delimiter=',')
		# #
		x, y = np.meshgrid(Data[:, 0], Data[:, 1])
		# # #z=Data[:,2]
		# # #z = z.reshape(-1, 1)
		# #
		fig = plt.figure(figsize=(18, 18))
		left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
		ax = fig.add_axes([left, bottom, width, height])
		cp = plt.contour(x, y, F.transpose(), inline=True)
		ax.clabel(cp, inline=True,
				  fontsize=8)
		#
		ax.set_title('Contour Plot for f')
		ax.set_xlabel('a')
		ax.set_ylabel('x')
		ax.set_xlim([-6.5, 6.5])
		ax.set_ylim([-0.32, 0.32])
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
				  fancybox=True, shadow=True, ncol=5)
		plt.savefig('contourplot_f.png', dpi=300)
		#
		# #
		fig = plt.figure(figsize=(18, 18))
		left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
		ax = fig.add_axes([left, bottom, width, height])
		cp = plt.contour(x, y, G.transpose(), inline=True)
		#
		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.1,
						 box.width, box.height * 0.9])
		#
		ax.clabel(cp, inline=True,
				  fontsize=10)
		#
		ax.set_title('Contour Plot for g')
		ax.set_xlabel('a')
		ax.set_ylabel('x')
		ax.set_xlim([-6.5, 6.5])
		ax.set_ylim([-0.32, 0.32])
		plt.savefig('contourplot_g.png', dpi=300)
	#plt.show()
	# x=2

	opt = Optimizer()
	x = opt.opt._x
	y = opt.opt._y

	if plot:
		n_points = y.shape[0]
		colours = {"S0": "green", "S1": "salmon", "S2": "DarkRed", "S3": "orange", "Boundary": "lightseagreen"}
		criterias = ["S0", "S1", "S2", "S3", "Boundary"]
		df = pd.DataFrame([[x[0, 0], x[0, 1], y[0, 0], y[0, 1], "S0", 0, 0]],
						  columns=['a', "x", "f", "g", "criteria", "boundary", "Fail"])
		for j in range(1, n_points):
			df2 = pd.DataFrame([[x[j, 0], x[j, 1], y[j, 0], y[j, 1], "S0", 0, 0]],
							   columns=['a', "x", "f", "g", "criteria", "boundary", "Fail"])
			df = df.append(df2)
	

	plt_indicator=0
	prev_plt_indicator=0

	Reward_data=np.zeros([9,2])
	j=0
	for i in range(num_experiments):
		if i%10==0:
			maximum, fval = opt.opt.get_maximum()
			Reward_data[j,1]=fval[0]
			Reward_data[j,0]=i
			j+=1
		if plot:
			df2 = df
			print(opt.opt.criterion)
			idx = df2["boundary"] > 0
			df2.criteria.loc[idx] = "Boundary"
			if i%30==0:
				for c in criterias:
					ix = np.where(df.criteria == c)[0]
					ax.scatter(df2.a.iloc[ix], df2.x.iloc[ix], c=colours[c], label=c if i == 0 else "")
				a_star = opt.opt.get_maximum()[0]
				max_point=ax.scatter(a_star, 0,s=50,facecolors='none',edgecolors="black",marker="s",label="Maximum" if i == 0 else "")
				ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5, title="criteria")
				name = 'contourplot_g' + str(i) + ".png"
				fig.savefig(name, dpi=300)
				max_point.remove()
				fig2 = plt.figure(figsize=(14, 10))
				left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
				ax2 = fig2.add_axes([left, bottom, width, height])

				x=np.zeros([100,2])
				a=np.linspace(-6,6,100)
				x[:,0]=a
				a = a.reshape(-1, 1)
				mean,var=opt.opt.gps[0].predict(x)
				std=np.sqrt(var)
				l_x0=mean-2*std
				u_x0=mean+2*std
				mean, var = opt.opt.gps[1].predict(x)
				std = np.sqrt(var)
				l_x0_1=mean-2*std
				u_x0_1=mean+2*std
				safe_a=a[np.logical_and(l_x0 >= opt.opt.fmin[0] ,l_x0_1 >=opt.opt.fmin[1])]

				f = (u_x0 + l_x0) / 2
				a = a.squeeze()
				safe=np.logical_and(l_x0 >= opt.opt.fmin[0] ,l_x0_1 >=opt.opt.fmin[1])
				data = np.vstack((a,f.squeeze(),safe.squeeze()))
				data=data.T
				np.savetxt("safeset_data.csv",data,delimiter=',')
				ax2.plot(a, f, color="black", label="f")


				ax2.fill_between(a,u_x0.squeeze(),l_x0.squeeze(), facecolor='blue',alpha=0.5)
				positive_a = safe_a[safe_a > 0]
				min_a_pos = np.min(positive_a)
				max_a_pos = np.max(positive_a)
				ax2.axvspan(min_a_pos, max_a_pos, color='green', alpha=0.5)
				negative_a = safe_a[safe_a < 0]
				min_a_neg = 0
				max_a_neg = 0
				if np.any(negative_a):
					min_a_neg = np.min(negative_a)
					max_a_neg = np.max(negative_a)
					ax2.axvspan(min_a_neg, max_a_neg, color="green", alpha=0.5)
				ax2.set_title('GP belief of objective function')
				ax2.set_xlabel('a')
				ax2.set_ylabel('f')
				ax2.set_xlim([-6.1, 5.1])
				ax2.set_ylim([-1000, 200])
				name = 'function_map' + str(i) + ".png"
				fig2.savefig(name, dpi=300)
				fig3 = plt.figure(figsize=(14, 10))
				left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
				ax3 = fig3.add_axes([left, bottom, width, height])
				l_x0 = l_x0_1
				u_x0 = u_x0_1
				g = (u_x0 + l_x0) / 2
				ax3.plot(a, g, color="black", label="g")
				ax3.fill_between(a,u_x0.squeeze(),l_x0.squeeze(), facecolor='blue',alpha=0.5)
				ax3.axvspan(min_a_pos, max_a_pos, color='green', alpha=0.5)
				if np.any(negative_a):
					ax3.axvspan(min_a_neg, max_a_neg, color="green", alpha=0.5)
					ax3.set_title('GP belief of constraint function')
					ax3.set_xlabel('a')
					ax3.set_ylabel('g')
				ax3.set_xlim([-6.1, 5.1])
				ax3.set_ylim([-2,2])
				name = 'function_map_g' + str(i) + ".png"
				fig3.savefig(name, dpi=300)

			df = df.append(opt.optimize())
		else:
			print(opt.opt.criterion)
			df=opt.optimize()
	np.savetxt("Rewards_gosafe.csv", Reward_data, delimiter=',')
	print(opt.opt.get_maximum())
	time_recorder=np.asarray(opt.time_recorded)
	print("Time:",time_recorder.mean(),time_recorder.std())





if __name__ == '__main__':
	start_time = time.time()
	Optimize()
	print("--- %s seconds ---" % (time.time() - start_time))

