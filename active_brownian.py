# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import animation
import seaborn as sns
import random


def brownian(N, dt, r, v, eta, T, W, L):
    kB = 1.38*1e-23 #Boltzmann constant [J/K]
    gamma = 6*np.pi*r*eta # friction coefficient [Ns/m]
    DT = kB*T/gamma # translational diffusion coefficient [m^2/s]
    DR = 6*DT/(8*r*r) # rotational diffusion coefficient [rad^2/s]
    x0 = random.uniform(-50*1e-6, 50*1e-6)
    y0 = random.uniform(-50*1e-6, 50*1e-6)
    theta0 = random.uniform(-3.14,3.14)
    n = 0   
    posx = []
    posy = []
#    print DT
#    print DR
    while n < N:
        theta = theta0 + np.sqrt(2*DR*dt)*norm.rvs(loc = 0, scale = 1) + dt*W
        x = x0 + np.sqrt(2*DT*dt)*norm.rvs(loc = 0, scale = 1) + dt*abs(x0)/(L/2)*v*np.cos(theta) 
        if abs(x) > L/2:
            d = abs(x)-L/2
            x = np.sign(x)*(L/2-d)
        y = y0 + np.sqrt(2*DT*dt)*norm.rvs(loc = 0, scale = 1) + dt*abs(x0)/(L/2)*v*np.sin(theta)
        if abs(y) > L/2:
            d = abs(y)-L/2
            y = np.sign(y)*(L/2-d)
        posx.append(x)
        posy.append(y)
        x0 = x
        y0 = y
        theta0 = theta
        n += 1
    posx1 = np.array(posx)*1e6
    posy1 = np.array(posy)*1e6
#    print posx[:10]
#    print posy[:10]
#    plt.plot(posx1, posy1)
#    plt.axis([-50, 50, -50, 50])
#    plt.show()
#    plt.ion()
#    for i, p in enumerate(posx1):
#        print p, posy1[i]
#        plt.scatter(p, posy1[i])
#        plt.draw()
#        plt.pause(0.1)
    return (posx1, posy1)


    
def mean_square_displacement(xvector,yvector):
    #Input: vector with 2d positions in tuples
    #Output: mean square displacement given by MSD(p) = sum( (r(i+p)-r(i))**2)/total
    length = len(xvector)
    intList = np.arange(1,length) #intervals
    MSD = np.arange(1,length, dtype = float) #To save the MSD values
    for interval in intList:
        intVector = [1]+[0]*(interval-1)+[-1] #Ex: for int = 3 you have [1,0,0,-1]
        #With "valid" you only take the overlapping points of the convolution
        convolutionx = np.convolve(xvector,intVector,'valid')
        convolutiony = np.convolve(yvector,intVector,'valid')
        MSDlist = convolutionx**2+convolutiony**2
        MSD[interval-1] = np.average(MSDlist)
    return intList,MSD
posx_collect = []    
posy_collect = []
q = 0
while q < 10000:
    p_x, p_y = brownian(1000, 0.01, 1*1e-6, 5*1e-6, 0.001, 300, 0, 100*1e-6)
    posx_collect.append(p_x)
    posy_collect.append(p_y)
    plt.axis([-60, 60, -60, 60])
    plt.plot(p_x, p_y)
    q += 1
plt.show()

posx_all = np.concatenate(posx_collect)
posy_all = np.concatenate(posy_collect)

#plt.hist2d(posx_all, posy_all, bins = 25, normed = True)
#plt.show()
plt.hist(posy_all, bins = 25, normed = True)
plt.show()


time, MSD = mean_square_displacement(p_x, p_y)
plt.plot(time, MSD)
#plt.axis([0.01, 1000, 0.01, 1000])
#plt.xscale('log')
#plt.yscale('log')

#def msd(posx, posy, t):
#    msd = []
#    for n in range(t):
#        for k in range(t-n):
#            loc = []
#            m = (posx[k+n]-posx[k])**2 + (posy[k+n]-posy[k])**2
#            loc.append(m)
#        a = np.array(loc)
#        av = np.average(a)
#        msd.append(av)
#    plt.plot( range(t), msd)    
##    plt.xlim([0,100])
##    plt.xscale('log')
##    plt.yscale('log')
#    return msd

    
#for i, p in enumerate(p_x):
##    print p, p_y[i]
#    plt.axis([-30, 30, -30, 30])
#    plt.scatter(p, p_y[i])
##    plt.draw()
#    plt.show()
#    plt.pause(0.00001)






#fig = plt.figure()
#ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50))
#line, = ax.plot([], [], lw=2)
#
#def init():
#    line.set_data([], [])
#    return line,
#    
#def animate(i):
#    p_x, p_y = brownian(1000, 0.01, 1*1e-6, 2*1e-6, 0.001, 300, 0)
#    line.set_data(p_x,p_y)
#    return line,
#    
#anim = animation.FuncAnimation(fig, animate, init_func=init,frames=50, interval=20, blit=True)
#
#plt.show() 

    
    
 
