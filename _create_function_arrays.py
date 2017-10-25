#!/usr/bin python2.7

import os
import matplotlib.pyplot as plt
import numpy as np
import math
__dir__ = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(__dir__, 'func.dat')
fobj = open(filepath, "w")

x = 0.1
y = 0.0

while x < 20.05:
    y = (math.exp(x))/(x**2)
    fobj.write(str(x))
    fobj.write('\t')
    fobj.write(str(y))
    fobj.write("\n")
    x = x + 0.1


fobj.close()


__dir__ = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(__dir__, 'func.dat')
fobj = open(filepath)
lines = fobj.readlines()
fobj.close()

x1 = []
y1 = []

for line in lines:
    p = line.split()
    x1.append(float(p[0]))
    y1.append(float(p[1]))

xv = np.array(x1)
yv = np.array(y1)

plt.plot(xv,yv, "b--")

#plt.axis([0, 10, 0, 300])

plt.show()
