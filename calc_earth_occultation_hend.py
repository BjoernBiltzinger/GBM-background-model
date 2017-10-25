import math
from math import cos, sin, atan, acos, fabs
import numpy
import matplotlib.pyplot as plt
import pylab
from tables import *
from pylab import *
from scipy import integrate
from work_module import calculate
from work_module import detector
from work_module import readfile
from work_module import writefile
calc = calculate()
det = detector()
rf = readfile()
wf = writefile()

sat_dist = 6912000.
r = 6383000.8

l = math.sqrt(sat_dist**2 + r**2)

day = 150926
detector = det.n5
poshist = rf.poshist_bin(day, 0, detector)
sat_time_bin = poshist[0]
sat_pos_bin = poshist[1]
sat_lat_bin = poshist[2]
sat_lon_bin = poshist[3]
bin_time_mid = poshist[5]
earth_ang = calc.earth_ang_bin(detector, day, bin_time_mid)[0]*2*math.pi/360.
#earth_ang = [0.896143067857]
opening_ang = 80. * pi /180.
i = 2398




Phi = 0.

print 'd / l = %e' % (sat_dist/l)

def a(Phi, d = sat_dist, l = l, earth_ang = earth_ang, i = i):

  return -4. * (d/l)**2. * (math.cos(Phi)**2. * math.sin(earth_ang[i])**2 + math.cos(earth_ang[i])**2)

def b(d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r):

  return cos(earth_ang[i]) * 4. * d / l * (1. + d**2. / l**2. - r**2. / l**2.)

def c(Phi, d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r):

  return 4. * (d/l)**2. * cos(Phi)**2 * sin(earth_ang[i])**2 - 1 - 2 * (d/l)**2 + 2. * (r/l)**2. - (d / l)**4. + 2 * (r/l)**2 * (d/l)**2. - (r/l)**4.

def theta0(Phi, d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r):
  
  D = (b()**2 - 4 * a(Phi) * c(Phi))
  print (-b()-sqrt(D))/(2.*a(Phi))
  if (D <= 1e-15):
    return 0.
  else:
    return min(acos((-b()-sqrt(D))/(2.*a(Phi))), opening_ang)
#  D = max(D, 0.)
#  if (fabs((-b()-sqrt(D))/(2.*a(Phi))) < 1.):
#    out = acos((-b()-sqrt(D))/(2.*a(Phi)))
#  else:
#    out = 0.

  #print ('theta0 = %e' % min(out, opening_ang))
#  return min(out, opening_ang)

def theta1(Phi, d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r):

  D = (b()**2 - 4 * a(Phi) * c(Phi))
  if (D <= 1e-15):
    return 0.
  else:
    return min(acos((-b()+sqrt(D))/(2.*a(Phi))), opening_ang)

#  D = max(D, 0.)
#  if (fabs((-b()+sqrt(D))/(2.*a(Phi))) < 1.):
#    out = acos((-b()+sqrt(D))/(2.*a(Phi)))
#  else:
#    out = 0.

  #print ('theta1 = %e' % min(out, opening_ang))
#  return min(out, opening_ang)

def f(Phi):

  return -cos(theta1(Phi)) + cos(theta0(Phi))

def f_below_horizon(Phi):

  if (Phi < 0.5 * pi):

    return -cos(theta1(Phi)) + 1.

  else:

    return -cos(theta0(Phi)) + 1.

#D = (b()**2 - 4 * a(Phi) * c(Phi))

print 'size of earth on the sky: %e' % atan(r/sat_dist)
print 'a = %e' % a(Phi)
print 'b = %e' % b()
print 'c = %e' % c(Phi)
print '-b/2a = %e' % (-b()/(2.*a(Phi)))
#print 'determinant D = %e' % D

'''philist = arange(0., pi * 1.009, 0.01 * (pi))
theta0list = empty(len(philist))
theta1list = empty(len(philist))

for i_phi in range(len(philist)):

  Phi = philist[i_phi]
  theta0list[i_phi] = theta0(Phi)
  theta1list[i_phi] = theta1(Phi)'''


#def phi1(d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r, opening_ang = opening_ang):

#            c_phi1 = (l**2 + d**2 - r**2 - 2*d*l*math.cos(opening_ang)*math.cos(earth_ang[i]))/(2*d*l*math.sin(opening_ang)*math.sin(earth_ang[i]))
#            p10 = math.acos(c_phi1)
#            p11 = math.sqrt(-2*d**2 *((2*math.cos(earth_ang[i])**2 -1)*l**2 + r**2) + d**4 + (l**2 - r**2)**2)/(2*d*l*math.sin(earth_ang[i]))
#            #p12 = -p12
#            p13 = math.pi /2.
#            #p14 = -p13
#            return min(p10, p11, p13)

#print 'our phi1 is %e' % phi1()
#exit()

area = integrate.quad(f_below_horizon, 0., pi)

print 'earth_ang: ' , earth_ang[i]
print 'spherical cap: ', pi * (1. - cos(atan(r/sat_dist)))
print 'area is ', area
#print theta0list
'''plt.plot(philist, theta0list)
plt.plot(philist, theta1list)
plt.plot(philist, theta1list - theta0list)
plt.axvline(pi/2.)
plt.draw()
plt.show()'''

#print (theta1list - theta0list)


#  def midnight(Phi, signum = 0):
#            if signum ==0:
##                val = (-b() + math.sqrt(b()**2 - 4*a(Phi)*c(Phi)))/(2*a(Phi))
# #           else:
#                val = (-b() - math.sqrt(b()**2 - 4*a(Phi)*c(Phi)))/(2*a(Phi))
#            return val
#        def c_theta(Phi, ind, opening_ang = opening_ang):
#            if ind ==0:
#                cos = min(mignight(Phi, 1), math.cos(opening_ang))
#            else:
#                cos = min(mignight(Phi, 0), math.cos(opening_ang))
#            return cos
#        def phi1(d = sat_dist, l = l, earth_ang = earth_ang, i = i, r = r, opening_ang = opening_ang):
#            c_phi1 = (l**2 + d**2 - r**2 - 2*d*l*math.cos(opening_ang)*math.cos(earth_ang[i]))/(2*d*l*math.sin(opening_ang)*math.sin(earth_ang[i]))
#            p10 = math.acos(c_phi1)
#            p11 = math.sqrt(-2*d**2 *((2*math.cos(earth_ang[i])**2 -1)*l**2 + r**2) + d**4 + (l**2 - r**2)**2)/(2*d*l*math.sin(earth_ang[i]))
#            #p12 = -p12
#            p13 = math.pi /2.
#            #p14 = -p13
#            return min(p10, p11, p13)
