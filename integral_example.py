import numpy as np
from scipy.integrate import quad,dblquad
import matplotlib.pyplot as plt

rx = np.random.normal(0.0,1.0)
ry = np.random.normal(0.0,1.0)

sx = 0.1

def likelihood(x,y):
  x += rx
  y += ry
  dx = (x/sx)
  dy = (y/sx)
  return np.exp(-0.5*(dx*dx+dy*dy))

def prior(x,y):
  return 1.0/(xmax-xmin)**2

def posterior(x,y):
  out = likelihood(x,y)*prior(x,y)
  return out

def prior1d(x):
  return 1.0/(xmax-xmin)

def posterior1d(x):
  out = likelihood(x,x)*prior1d(x)
  return out

xmin = -10.0
xmax = 10.0
x,y = np.linspace(xmin,xmax,256),np.linspace(xmin,xmax,256)
X,Y = np.meshgrid(x,y)

print "2d evidence:",dblquad(posterior,xmin, xmax,lambda x: xmin, lambda x: xmax)
print "1d evidence:",quad(posterior1d, xmin, xmax)

plt.figure()
plt.contourf(X,Y,posterior(X,Y),100)
plt.colorbar()
plt.plot(x,y,'k')
plt.show()