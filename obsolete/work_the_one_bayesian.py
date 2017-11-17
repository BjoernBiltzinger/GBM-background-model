#!/usr/bin python2.7

import math
from   datetime import datetime

import numpy                       as np
import pymc3                       as pm
import scipy.optimize              as optimize
import theano
import theano.tensor               as T

from gbmbkgpy.utils.external_prop import ExternalProps, writefile
from   gbmbkgpy.work_module_refactor import calculate
from   gbmbkgpy.work_module_refactor import detector

calc = calculate()
det = detector()
rf = ExternalProps()
wf = writefile()

#docstrings of the different self-made classes within the self-made module
#cdoc = calc.__doc__
#ddoc = det.__doc__
#rdoc = rf.__doc__


##define the day, detector, datatype and energy channel one wants to look at##
day                    = 150926
detector               = det.n5
data_type              = 'ctime'
year                   = int('20' + str(day)[0:2])
echan                  = 1

#get the iso-date-format from the day
date                   = datetime(year, int(str(day)[2:4]), int(str(day)[4:6]))


##read the measurement data##
ctime_data             = rf.ctime(detector, day)
energy_chans           = ctime_data[0]
total_counts           = ctime_data[1]
echan_counts           = ctime_data[2]
total_rate             = ctime_data[3]
echan_rate             = ctime_data[4]
bin_time               = ctime_data[5]
good_time              = ctime_data[6]
exptime                = ctime_data[7]
bin_time_mid           = np.array((bin_time[:,0]+bin_time[:,1])/2)

#define which count-rate to look at
counts                 = echan_rate[echan]
#counts                 = total_rate


##read the satellite data##
sat_data               = rf.poshist_bin(day, bin_time_mid, detector, data_type)
sat_time_bin           = sat_data[0]
sat_pos_bin            = sat_data[1]
sat_lat_bin            = sat_data[2]
sat_lon_bin            = sat_data[3]
sat_q_bin              = sat_data[4]

sat_time               = rf.poshist(day)[0]#satellite time with onboard binning, not the measurement binning (needed for the mcilwain interpolation)


#read the SFL data
flares                 = rf.flares(year)
flares_day             = flares[0]
flares_time            = flares[1]
if np.any(flares_day == day) == True:
    flares_today           = flares_time[:,np.where(flares_day == day)]
    flares_today           = np.squeeze(flares_today, axis=(1,))/3600.
else:
    flares_today           = np.array(-5)


##get the fitting functions##
#calculate the sun data
sun_data               = calc.sun_ang_bin(detector, day, bin_time_mid, data_type)
sun_ang_bin            = sun_data[0]
sun_ang_bin            = calc.ang_eff(sun_ang_bin, echan)[0]

#calculate the earth data
earth_data             = calc.earth_ang_bin(detector, day, bin_time_mid, data_type)
earth_ang_bin          = earth_data[0]
#earth_ang_bin          = calc.ang_eff(earth_ang_bin, echan)[0]
earth_ang_bin          = calc.earth_occ_eff(earth_ang_bin, echan)

#get the mcilwain parameter as a representative for the magnetic field
lat_data               = rf.mcilwain(day)
mc_b                   = lat_data[1]
mc_l                   = lat_data[2]
mc_b                   = calc.intpol(mc_b, day, 0, sat_time, bin_time_mid)[0]
mc_l                   = calc.intpol(mc_l, day, 0, sat_time, bin_time_mid)[0]
magnetic               = mc_l #define which mcilwain parameter to look at -> L-parameter as default

#constant function corresponding to the diffuse y-ray background or constant background noise (interpretation slightly off, when the detector is occulted by the earth)
cgb                    = np.ones(len(counts))


##begin bayesian fitting##
#make fitting functions zero when the detectors are turned off during the saa
zeroCon                = counts==0
cgb[zeroCon]           = 0
earth_ang_bin[zeroCon] = 0
sun_ang_bin[zeroCon]   = 0
magnetic[zeroCon]      = 0


#define parameters in c++ for the model to make the code faster
background_model       = pm.Model()

shared_cgb             = theano.shared(cgb[~zeroCon])
shared_magnetic        = theano.shared(magnetic[~zeroCon])
shared_earth_ang_bin   = theano.shared(earth_ang_bin[~zeroCon])
shared_sun_ang_bin     = theano.shared(sun_ang_bin[~zeroCon])
shared_counts          = theano.shared(counts[~zeroCon])
shared_bin_time_mid    = theano.shared(bin_time_mid[~zeroCon])

#define the background-model
with background_model:
    
    #define the parameter range
    a_log                  = pm.Uniform("a_log",lower=math.log10(10.),upper=math.log10(200))
    b_log                  = pm.Uniform("b_log",lower=math.log10(10.),upper=math.log10(200))
    a                      = pm.Deterministic('a',T.pow(10.,a_log))
    b                      = pm.Deterministic('b',T.pow(10.,b_log))
    #a                      = pm.Uniform("a",lower=10.,upper=200)
    #b                      = pm.Uniform("b",lower=10.,upper=200)
    c                      = pm.Uniform("c",lower=-20,upper=20)
    d                      = pm.Uniform("d",lower=-20,upper=20)
    
    #define the fitting function
    combined_curve         = pm.Deterministic("combined_curve",a + b*shared_magnetic + c*shared_earth_ang_bin + d*shared_sun_ang_bin )
    
    #define the statistics behind the measured data (Poisson)
    likelihood             = pm.Poisson("likelihood",mu=combined_curve,observed=shared_counts)


with background_model:
    
    start                  = pm.find_MAP(fmin=optimize.fmin_powell )


#define the number of steps within the parameter interval
n_samples              = 2500
burn_in                = 500

with background_model:
    
    #define steps of the bayesian fit (NUTS is most accurate, but takes longer to calculate)
    step                   = pm.NUTS(scaling=start)   
    #step                   = pm.Metropolis()
    
    #start fitting calculations
    samples                = pm.sample(n_samples,step=step,start=start,progressbar=True)


#see the fitting calculations' results
#pm.traceplot(samples[burn_in:],varnames=['a','b','c','d’])
#pm.summary(samples[burn_in:],varnames=['a','b','c','d’])


#####plot-algorhythm#####
#convert the x-axis into hours of the day
plot_time_bin_date = calc.met_to_date(bin_time_mid)[0]
plot_time_bin = (plot_time_bin_date - calc.day_to_met(day)[1])*24#Time of day in hours
plot_time_sat_date = calc.met_to_date(sat_time_bin)[0]
plot_time_sat = (plot_time_sat_date - calc.day_to_met(day)[1])*24#Time of day in hours


#fit the background-function and the seperate functions to the data
fit_curve_all          = []
thin                   = 10

for tr in samples[burn_in::thin]:
    
    fit_curve              = tr['a']*cgb + tr['b']*magnetic + tr['c']*earth_ang_bin + tr['d']*sun_ang_bin
    plot(plot_time_bin,fit_curve,'r',alpha=.01)
    
    fit_curve              = tr['a']*cgb
    plot(plot_time_sat,fit_curve,'k',alpha=.01)
    
    fit_curve              = tr['b']*magnetic
    plot(plot_time_sat,fit_curve,'g',alpha=.01)
    
    fit_curve              = tr['c']*earth_ang_bin
    plot(plot_time_sat,fit_curve,'c',alpha=.01)
    
    fit_curve              = tr['d']*sun_ang_bin
    plot(plot_time_sat,fit_curve,'y',alpha=.01)

#plot vertical lines for the solar flares of the day
if np.all(flares_today != -5):
    if len(flares_today[0]) > 1:
        for i in range(0, len(flares_today[0])):
            plot(x = flares_today[0,i], ymin = 0., ymax = 1., linewidth=2, color = 'grey')
            plot(x = flares_today[1,i], ymin = 0., ymax = 1., color = 'grey', linestyle = '--')
    else:
        plot(x = flares_today[0], ymin = 0., ymax = 1., linewidth=2, color = 'grey')
        plot(x = flares_today[1], ymin = 0., ymax = 1., color = 'grey', linestyle = '--')



###plot each on the same axis as converted to counts###
'''fig, ax1 = plt.subplots()

plot1 = ax1.plot(plot_time_bin, counts, 'b-', label = 'Countrate')
plot2 = ax1.plot(plot_time_bin, fit_curve, 'r-', label = 'Fit')
plot3 = ax1.plot(plot_time_sat, d*sun_ang_bin, 'y-', label = 'Sun angle')
plot4 = ax1.plot(plot_time_sat, c*earth_ang_bin, 'c-', label = 'Earth angle')
plot5 = ax1.plot(plot_time_sat, b*magnetic, 'g-', label = 'Magnetic field')
plot6 = ax1.plot(plot_time_sat, a*cgb, 'b--', label = 'Cosmic y-ray background')
#plot7 = ax1.plot(plot_time_sat, j2000_orb, 'y--', label = 'J2000 orbit')
#plot8 = ax1.plot(plot_time_sat, geo_orb, 'g--', label = 'Geographical orbit')

plots = plot1 + plot2 + plot3 + plot4 + plot5 + plot6# + plot7 + plot8
labels = [l.get_label() for l in plots]
ax1.legend(plots, labels, loc=1)

ax1.grid()

ax1.set_xlabel('Time of day in 24h')
ax1.set_ylabel('Countrate')

#ax1.set_xlim([9.84, 9.85])
ax1.set_xlim([-0.5, 24.5])
ax1.set_ylim([-200, 2000])

#get the ordinal indicator for the date
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])

plt.title(data_type + '-countrate-fit of the ' + detector.__name__ + '-detector on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(year))

plt.show()'''


###plot residual noise of the fitting algorithm###
'''plt.plot(plot_time_bin, counts - fit_curve, 'b-')

plt.xlabel('Time of day in 24h')
plt.ylabel('Residual noise')

plt.grid()

plt.title(data_type + '-counts-fit residuals of the ' + detector.__name__ + '-detector on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(year))

plt.ylim([-200, 200])

plt.show()'''

