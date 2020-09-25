import numpy as np
import math
import matplotlib.pyplot as plt

def harmonic(x,x0=0,k=1.0):
    pe = 0.5*k*(x-x0)**2
    return pe
def der_harmonic(x,x0,k=1.0):
    f = -k*(x-x0)
    return f

def produceTrajectory(numTrajectories, width, potentialFunction, filename, 
        start=0, dt=0.01, T=1, k=10, 
        R=0.001, xc=0, K=0, k_=0, damping=1, beta=1): #potential function - write to filename
#vectorized production of trajectories as per langevin dynamics.
#Euler Maruyama method
#Arguments:#
#numTrajectories - number of trajectories desired
#width - this is a holder, helps separate between the potentials with width 2 and width 4 for generating random starting areas
#potentialFunction - this is the potDerBS or potDerABS above.  Should be a derivative function
#filename - output filename for data generated
#start - start time
#dt - timestep
#T - end time
#k - force constant for potential
#R - 
#xc - 
#K - 
#k_ -
#damping - 1 is overdamped langevin
#beta - thermodynamic beta
    if filename != "test":
        f = open(filename, "a")
    #vectorized
    g = math.sqrt(2/(beta*dt*damping))
    t=0
    print(dt)
    np.random.seed(99)
    _R = np.random.RandomState(numTrajectories)
    randomSampling = _R.random_sample(numTrajectories)#random numbers from 0-1
    X = 40 #x_center + randomSampling * 4
    v = 0.
    traj = np.zeros((int(T/dt)+2, numTrajectories))
    deviateAvailable = False
    step=0
    while (t<T+dt):
        #randomd, deviateAvailable = randn(0,1,deviateAvailable)
        randomd = np.random.normal(loc=0, scale=1, size=numTrajectories)
        v = g * randomd + potentialFunction(X) / damping
        X += dt * v
        traj[step] = X
        step = step + 1
        t += dt
    if filename != "test":
        for i in traj:
            f.write(str(i))
            f.write("\n")
        f.close()
    return np.transpose(traj)
beta = 0.1
x_center = 45.
step_size = 1e-4
total_time = 4.
dh1 = lambda x: der_harmonic(x,x_center,2)
traj1 = (produceTrajectory(4, 2, dh1,"test", dt=step_size, T=total_time))
print(traj1)
time = np.arange(0,total_time+step_size,step_size)

plt.plot(time,traj1[0,:-1],label="Trajectory 1")
plt.plot(time,traj1[1,:-1],label="Trajectory 2")
plt.plot(time,traj1[2,:-1],label="Trajectory 3")
plt.plot(time,traj1[3,:-1],label="Trajectory 4")
plt.title(r"$r_c={0} \AA$".format(x_center))
plt.xlabel("Time, seconds")
plt.ylabel(r"Dye-Dye Distance, $\AA$")
plt.show()

plt.subplot(211)
bins = np.linspace(0,90,100)
plt.hist(traj1[0],bins,density=True,label="Trajectory 1",alpha=0.5)
plt.hist(traj1[1],bins,density=True,label="Trajectory 2",alpha=0.5)
plt.hist(traj1[2],bins,density=True,label="Trajectory 3",alpha=0.5)
plt.hist(traj1[3],bins,density=True,label="Trajectory 4",alpha=0.5)
plt.xlabel(r"Dye-Dye Distance, $\AA$")
plt.ylabel("Density")
plt.legend()

def efficiency(r,r0):
    return 1/(1+(r/r0)**6)

E = efficiency(traj1,38.)
bins = np.linspace(0,1,50)
plt.subplot(212)
plt.hist(E[0],bins,density=True,label="Trajectory 1",alpha=0.5)
plt.hist(E[1],bins,density=True,label="Trajectory 2",alpha=0.5)
plt.hist(E[2],bins,density=True,label="Trajectory 3",alpha=0.5)
plt.hist(E[3],bins,density=True,label="Trajectory 4",alpha=0.5)
plt.xlabel("Efficiency")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
