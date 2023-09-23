# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 01:01:33 2023

@author: MiloPC
"""

import torch

import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_device('cpu')



class Narm:
    def __init__(self,N):
            
        self.N = N        
        self.setParams()
        self.setIC()
        # self.setIC()

        self.cfg = []

        for n in range(self.N):
                        #     name             wgt      lower        upper
            self.cfg +=   [['\u03B8'+str(n+1),       1.,   -torch.pi/4,   torch.pi/4]]
        
        self.cfg[0][1] = 0.0
        
    def reset(self):
        self.__init__(self.N)
        
    def setParams(self):
        self.thetas = torch.pi*(2*torch.rand([self.N,1],requires_grad=True)-1)
        self.joint_lengths = torch.ones(self.N)/self.N
        self.xs = torch.zeros(self.N+1)
        self.ys = torch.zeros(self.N+1)
        self.xt = torch.tensor(0.5)
        self.yt = torch.tensor(0.5)
    def setIC(self):
        self.vars = torch.tensor(self.thetas)
        
    def setVars(self,X):
        self.thetas = X.clone()
        
    def forward(self,X):
        for s in range(self.N):
            self.xs[s+1] = self.xs[s] + self.joint_lengths[s]*torch.cos(torch.sum(X[0:s+1]))
            self.ys[s+1] = self.ys[s] + self.joint_lengths[s]*torch.sin(torch.sum(X[0:s+1]))
            
        self.prod = X.prod()
            
        self.res = self.resFun()
    
        return self.res
        
    def resFun(self):
        return torch.cat([(self.xs[-1] - self.xt).view(-1), (self.ys[-1] - self.yt).view(-1), self.yt.view(-1)],dim=0)
 
    def plot_init(self):
        
        # plt.close('all') 
        
        self.fig = plt.figure(figsize=plt.figaspect(1))
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.p1, = self.ax.plot(self.xs.detach().cpu().numpy(),self.ys.detach().cpu().numpy(),'b')
        self.p2, = self.ax.plot(self.xt.detach().cpu().numpy(),self.yt.detach().cpu().numpy(),'rx')

        self.ax.set_xlim(-1.,1.)
        self.ax.set_ylim(-1.,1.)
        self.ax.set_aspect('equal')
        
    def plot_update(self):
        
        self.p1.set_xdata(self.xs.detach().cpu().numpy())
        self.p1.set_ydata(self.ys.detach().cpu().numpy())
        self.p2.set_xdata(self.xt.detach().cpu().numpy())
        self.p2.set_ydata(self.yt.detach().cpu().numpy())

        self.fig.canvas.draw()
            
        
        
        
        
        
class SimplePlane: 
    
    def resFun(self):
        # return (self.tau, self.Fy, self.CoMx, self.CoPx)
        return torch.cat([self.tau.view(-1), self.Fy.view(-1), self.CoMx.view(-1), self.mtot.view(-1)],dim=0)
        # return (self.tau, self.Fy)
    
    def setIC(self):
        self.vars = torch.tensor([[self.A1],
                                  [self.A2],
                                  [self.x1],
                                  [self.x2],
                                  [self.x3],
                                  [self.theta1],
                                  [self.theta2],
                                  [self.m3],
                                  [self.rhoA],
                                  [self.rhoL]])
        
    def setParams(self):
       self.g = torch.tensor(-9.8,requires_grad=True)
       self.V_inf = torch.tensor(10.0,requires_grad=True)


       self.theta_inf = torch.tensor(0.0,requires_grad=True)
       self.theta = torch.tensor(0.0,requires_grad=True)
       self.Vt = self.V_inf*torch.tensor([[torch.cos(self.theta_inf)],
                                          [torch.sin(self.theta_inf)]],requires_grad=True)
       self.rho = torch.tensor(1.225,requires_grad=True)
       self.rhoA = torch.tensor(0.3,requires_grad=True)
       self.rhoL = torch.tensor(0.01,requires_grad=True)
       self.omega = torch.tensor(0.0,requires_grad=True)

       self.x1 = torch.tensor(0.0,requires_grad=True)
       self.ar1 = torch.tensor(10.0,requires_grad=True)
       self.A1 = torch.tensor(0.06,requires_grad=True)
       self.theta1 = torch.tensor(-2.0*torch.pi/180,requires_grad=True)
       self.CLa1 = torch.tensor(5.0,requires_grad=True)

       self.x2 = torch.tensor(0.25,requires_grad=True)
       self.ar2 = torch.tensor(5.0,requires_grad=True)
       self.A2 = torch.tensor(0.01,requires_grad=True)
       self.theta2 = torch.tensor(2.*torch.pi/180,requires_grad=True)
       self.CLa2 = torch.tensor(5.0,requires_grad=True)

       self.x3 = torch.tensor(-.1,requires_grad=True)
       self.m3 = torch.tensor(0.005,requires_grad=True)
       
       self.k = 2/(torch.tensor(torch.pi)).sqrt()    
    
    def setVars(self,X):
        self.A1 =      (X[0].clone()**2).sqrt()
        self.A2 =      (X[1].clone()**2).sqrt()
        self.x1 =       X[2].clone()
        self.x2 =       X[3].clone()
        self.x3 =       X[4].clone()
        self.theta1 =   X[5].clone()
        self.theta2 =   X[6].clone()
        self.m3 =      (X[7].clone()**2).sqrt()
        self.rhoA =    (X[8].clone()**2).sqrt()
        self.rhoL =    (X[9].clone()**2).sqrt()
    
    def __init__(self):
        
        self.setParams()
        self.setIC()

                    #   name              wgt      lower      upper
        self.cfg =   [['Area Wing',      0.25,   2.27e-02,   0.0377],
                      ['Area Tail',      0.25,   1.00e-04,   0.0039],
                      ['Pos Wing',       0.00,  -1.06e-01,   0.1438],
                      ['Pos Tail',       0.00,   0.0,        0.6000],
                      ['Pos Nose',       0.00,  -0.25,       0.0000],
                      ['\u03B1 Wing',    0.25,  -1.0e-1,   1.0e-1],
                      ['\u03B1 Tail',    0.25,  -1.0e-1,   1.0e-1],
                      ['Mass Nose',      0.00,   2.00e-03,   0.0420],
                      ['\u03C1 Area',    0.00,   5.00e-02,   0.9500],
                      ['\u03C1 Length',  0.00,   1.00e-03,   0.0190]]

    def reset(self):
        
        self.__init__()         
    
    def plot_init(self):
        
        # plt.close('all') 
        
        self.vec = torch.linspace(-0.5,0.5,2)
        self.x,self.y = torch.meshgrid(self.vec,self.vec)

        self.x1plo = (self.x*self.b1 + self.x1).detach().cpu().numpy().reshape([-1])[[0,2,3,1]]
        self.y1plo = (self.y*self.s1).detach().cpu().numpy().reshape([-1])[[0,2,3,1]]
        self.x2plo = (self.x*self.b2 + self.x2).detach().cpu().numpy().reshape([-1])[[0,2,3,1]]
        self.y2plo = (self.y*self.s2).detach().cpu().numpy().reshape([-1])[[0,2,3,1]]
        
        self.x1plo = np.concatenate((self.x1plo, self.x1plo[0].reshape([-1])))
        self.y1plo = np.concatenate((self.y1plo, self.y1plo[0].reshape([-1])))
        self.x2plo = np.concatenate((self.x2plo, self.x2plo[0].reshape([-1])))
        self.y2plo = np.concatenate((self.y2plo, self.y2plo[0].reshape([-1])))

        self.fig = plt.figure(figsize=plt.figaspect(1))
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        self.wire1, = self.ax.plot(self.x1plo,self.y1plo,self.x1plo*0.0, 'r',linewidth = 1)
        self.wire2, = self.ax.plot(self.x2plo,self.y2plo,self.x2plo*0.0, 'b',linewidth = 1)
        self.line1, = self.ax.plot(np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]), 'm',linewidth = 2)
        self.p1, = self.ax.plot(0.,0.,0.,'bo')
        self.p2, = self.ax.plot(self.CoPx.detach().cpu().numpy(),0.,0.,'ro')
        self.p3, = self.ax.plot(self.CoMx.detach().cpu().numpy(),0.,0.,'go')
        self.ax.set_xlim(-0.5,0.5)
        self.ax.set_ylim(-0.5,0.5)
        self.ax.set_aspect('equal')
        
        self.xplon = 0.5*torch.cos(torch.linspace(0.0,2*torch.pi,11))
        self.yplon = 0.5*torch.sin(torch.linspace(0.0,2*torch.pi,11))
        
    def plot_update(self):

        self.x1plo = (self.xplon*self.b1 + self.x1).detach().cpu().numpy().reshape([-1])
        self.y1plo = (self.yplon*self.s1).detach().cpu().numpy().reshape([-1])
        self.x2plo = (self.xplon*self.b2 + self.x2).detach().cpu().numpy().reshape([-1])
        self.y2plo = (self.yplon*self.s2).detach().cpu().numpy().reshape([-1])
        
        self.x1plo = np.concatenate((self.x1plo, self.x1plo[0].reshape([-1])))
        self.y1plo = np.concatenate((self.y1plo, self.y1plo[0].reshape([-1])))
        self.x2plo = np.concatenate((self.x2plo, self.x2plo[0].reshape([-1])))
        self.y2plo = np.concatenate((self.y2plo, self.y2plo[0].reshape([-1])))
        
        self.wire1.set_data_3d(self.x1plo,self.y1plo,self.y1plo*0)
        self.wire2.set_data_3d(self.x2plo,self.y2plo,np.abs(self.y2plo))           
        
                             
        self.line1.set_data_3d(np.array([self.x2.squeeze().detach().cpu().numpy(),self.x3.squeeze().detach().cpu().numpy()]),np.array([0.0,0.0]),np.array([0.0,0.0]))
        self.p1.set_data_3d(np.array([self.x3.squeeze().detach().cpu().numpy()]),np.array([0.0]),np.array([0.0]))
        self.p2.set_data_3d(self.CoPx.detach().cpu().numpy(),np.array([0.0]),np.array([0.0]))
        self.p3.set_data_3d(self.CoMx.detach().cpu().numpy(),np.array([0.0]),np.array([0.0]))
    
        self.fig.canvas.draw()

    def forward(self, X):
    
        self.setVars(X)
        
        self.b1 = (self.A1/self.ar1).sqrt()*self.k
        self.b2 = (self.A2/self.ar2).sqrt()*self.k

        self.s1 = self.A1/self.b1*self.k
        self.s2 = self.A2/self.b2*self.k
        
        self.Vt = self.V_inf*torch.tensor([[torch.cos(self.theta_inf)],
                                           [torch.sin(self.theta_inf)]])
        
        self.m1 = self.A1*self.rhoA
        self.m2 = self.A2*self.rhoA
        
        self.m4 = self.rhoL*((self.x2-self.x3)**2).sqrt()
        self.x4 = (self.x2-self.x3)*0.5
        
        
        self.mtot = self.m1+self.m2+self.m3+self.m4
        
        self.CoMx = (self.m1*self.x1 + self.m2*self.x2 + self.m3*self.x3 + self.m4*self.x4)/(self.mtot)
        
        self.r1 = self.x1 - self.CoMx
        self.r2 = self.x2 - self.CoMx
        
        self.V1 = self.Vt + self.omega*self.r1*torch.tensor([[-torch.sin(self.theta)],
                                          [-torch.cos(self.theta)]])
        
        self.V2 = self.Vt + self.omega*self.r2*torch.tensor([[-torch.sin(self.theta)],
                                          [-torch.cos(self.theta)]])
        
        self.alpha1 = torch.atan2(self.V1[1],self.V1[0]) + self.theta1
        self.alpha2 = torch.atan2(self.V2[1],self.V2[0]) + self.theta2
        
        self.CL1 = self.CLa1*self.alpha1
        self.CL2 = self.CLa2*self.alpha2
        
        self.lift1 = 0.5*self.CL1*self.A1*self.rho*(self.V1**2).sum()
        self.lift2 = 0.5*self.CL2*self.A2*self.rho*(self.V2**2).sum()
        
        self.Lift = self.lift1 + self.lift2
        
        self.CoPx = (self.lift1*(self.x1 - self.b1/4) + self.lift2*(self.x2 - self.b2/4))/self.Lift
        
        self.Fy = self.Lift + self.mtot*self.g
        
        self.tau1 = self.lift1*(self.r1 - self.b1/4)
        self.tau2 = self.lift2*(self.r2 - self.b2/4)
        
        
        # self.drag1 = self.alpha1**2
        
        # self.drag = self.drag1 + self.drag2
        
        self.tau = self.tau1 + self.tau2
        
        self.CMP_diff = self.CoPx - self.CoMx
        
        self.res = self.resFun()
        
        return self.res
        