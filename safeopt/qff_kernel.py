import numpy as np
from paramz import ObsAr
import torch
class QFF_process(object):
    def __init__(self,embedding,sigma):

        self.embedding=embedding
        self.theta_mean=np.zeros(self.embedding.m)
        self.theta_std=np.eye([self.embedding.m,self.embedding.m])
        self.theta_std_inv=self.theta_std
        self.sigma=sigma
        self.X=np.empty(1)
        self.Y=np.empty(1)

    def predict_noiseless(self,x):
        x=torch.from_numpy(x)
        Phi=self.embedding.embed(x)
        Phi=Phi.numpy()
        mu=np.dot(Phi,self.theta_mean.reshape(-1,1))
        var=self.sigma**2 * np.dot(Phi.T,np.dot(self.theta_std_inv,Phi))
        return mu,var

    def set_XY(self,X,Y):
        self.update_weights(X,Y)
        #X=torch.from_numpy(X)
        #Y=torch.from_numpy(Y)
        self.X=X
        self.Y=Y

    def update_weights(self,X,Y):
        Phi=self.embedding.embed(torch.from_numpy(X))
        Phi=Phi.nump()
        self.theta_std= np.dot(Phi.T,Phi) + np.eye(X.shape())*self.sigma**2
        c = np.linalg.inv(np.linalg.cholesky(self.theta_std))
        self.theta_std_inv=np.dot(c.T,c)
        self.theta_mean=np.dot(self.theta_std_inv,np.dot(Phi.T,Y))


    