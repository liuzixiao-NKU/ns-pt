from gcp import *
from wishart import *
from pylab import *

D = Wishart(2)
D.setDof(4)
for _ in xrange(10):
  a = D.sample()
  matshow(a)
  colorbar()
  show()