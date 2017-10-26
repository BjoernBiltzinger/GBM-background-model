    def curve_fit_plots(self, day, detector, echan, data_type = 'ctime'):
        """This function calculates the chi-square fit to the data of a given detector, day and energy channel and saves the figure in the appropriate folder\n
        Input:\n
        calculate.curve_fit_plots( day = YYMMDD, detector (f. e. n5), echan (9 or 129 = total counts for ctime or cspec), data_type = 'ctime' (or 'cspec'))\n
        Output:\n
        None"""
        
        detector = detector.detector
        year = int('20' + str(day)[0:2])
        
        #get the iso-date-format from the day
        date = datetime(year, int(str(day)[2:4]), int(str(day)[4:6]))
        
        #get the ordinal indicator for the date
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
        
        #read the measurement data
        ctime_data = readfile.ctime(readfile(), detector, day)
        echan = ctime_data[0]
        total_counts = ctime_data[1]
        echan_counts = ctime_data[2]
        total_rate = ctime_data[3]
        echan_rate = ctime_data[4]
        bin_time = ctime_data[5]
        good_time = ctime_data[6]
        exptime = ctime_data[7]
        bin_time_mid = np.array((bin_time[:,0]+bin_time[:,1])/2)
        
        total_rate = np.sum(echan_rate[1:-2], axis = 0)
        
        if data_type == 'ctime':
            if echan < 9:
                counts = echan_rate[echan]
            elif echan == 9:
                counts = total_rate
                echan = 0
            else:
                print 'Invalid value for the energy channel of this data type (ctime). Please insert an integer between 0 and 9.'
                return
        elif data_type == 'cspec':
            if echan < 129:
                counts = echan_rate[echan]
            elif echan == 129:
                counts = total_rate
                echan = 0
            else:
                print 'Invalid value for the energy channel of this data type (cspec). Please insert an integer between 0 and 129.'
                return
        else:
            print 'Invalid data type: ' + str(data_type) + '\n Please insert an appropriate data type (ctime or cspec).'
            return
                
        #read the satellite data
        sat_data = readfile.poshist_bin(readfile(), day, bin_time_mid, detector, data_type)
        sat_time_bin = sat_data[0]
        sat_pos_bin = sat_data[1]
        sat_lat_bin = sat_data[2]
        sat_lon_bin = sat_data[3]
        sat_q_bin = sat_data[4]
        
        #calculate the sun data
        sun_data = calculate.sun_ang_bin(calculate(), detector, day, bin_time_mid, data_type)
        sun_ang_bin = sun_data[0]
        sun_ang_bin = calculate.ang_eff(calculate(), sun_ang_bin, echan)[0]
        sun_rad = sun_data[2]
        sun_ra = sun_rad[:,0]
        sun_dec = sun_rad[:,1]
        sun_occ = calculate.src_occultation_bin(calculate(), day, sun_ra, sun_dec, bin_time_mid)[0]
        
        #calculate the earth data
        earth_data = calculate.earth_ang_bin(calculate(), detector, day, bin_time_mid, data_type)
        earth_ang_bin = earth_data[0]
        #earth_ang_bin = calc.ang_eff(earth_ang_bin, echan)[0]
        earth_ang_bin = calculate.earth_occ_eff(calculate(), earth_ang_bin, echan)
        
        #read the SFL data
        flares = readfile.flares(readfile(), year)
        flares_day = flares[0]
        flares_time = flares[1]
        if np.any(flares_day == day) == True:
            flares_today = flares_time[:,np.where(flares_day == day)]
            flares_today = np.squeeze(flares_today, axis=(1,))/3600.
        else:
            flares_today = np.array(-5)
        
        #read the mcilwain parameter data
        sat_time = readfile.poshist(readfile(), day)[0]
        lat_data = readfile.mcilwain(readfile(), day)
        mc_b = lat_data[1]
        mc_l = lat_data[2]
        
        mc_b = calculate.intpol(calculate(), mc_b, day, 0, sat_time, bin_time_mid)[0]
        mc_l = calculate.intpol(calculate(), mc_l, day, 0, sat_time, bin_time_mid)[0]
        
        magnetic = mc_l
        magnetic = magnetic - np.mean(magnetic, dtype=np.float64)
        
        #constant function corresponding to the diffuse y-ray background
        cgb = np.ones(len(total_rate))
        
        #counts[120000:] = 0
        cgb[np.where(total_rate == 0)] = 0
        earth_ang_bin[np.where(total_rate == 0)] = 0
        sun_ang_bin[np.where(sun_occ == 0)] = 0
        sun_ang_bin[np.where(total_rate == 0)] = 0
        magnetic[np.where(total_rate == 0)] = 0
        
        #remove vertical movement from scaling sun_ang_bin
        sun_ang_bin[sun_ang_bin>0] = sun_ang_bin[sun_ang_bin>0] - np.min(sun_ang_bin[sun_ang_bin>0])
        
        
        
        
        
        
        saa_exits = [0]
        for i in range(1, len(total_rate)):
            if np.logical_and(total_rate[i-1] == 0, total_rate[i] != 0):
                #print i
                saa_exits.append(i)
        saa_exits = np.array(saa_exits)
        
        if saa_exits[1] - saa_exits[0] < 10:
            saa_exits = np.delete(saa_exits, 0)
        
        
        
        
        
        
        
        
        def exp_func(x, a, b, i, addition):
        #    if saa_exits[i] == 0:
        #        addition = 0
        #    elif saa_exits[i] < 200:
        #        addition = 10
        #    else:
        #        addition = 9
        #    for i in range(0, 30):
        #        if i > addition:
        #            addition = i
        #            break
        #    print addition
        #    addition = round(addition)
            x_func = x[saa_exits[i]+math.fabs(addition):] - x[saa_exits[i]+math.fabs(addition)]
            func = math.fabs(a)*np.exp(-math.fabs(b)*x_func)
            zeros = np.zeros(len(x))
            zeros[saa_exits[i]+math.fabs(addition):] = func
            zeros[np.where(total_rate==0)] = 0
            return zeros
        
        
        
        
        
        
        deaths = len(saa_exits)
        
        exp = []
        for i in range(0, deaths):
            exp = np.append(exp, [40., 0.001])
        
        
        
        
        
        
        
        def fit_function(x, a, b, c, d, addition, exp1, exp2, deaths):
            this_is_it = a*cgb + b*magnetic + c*earth_ang_bin + d*sun_ang_bin
            for i in range(0, deaths):
                this_is_it = np.add(this_is_it, exp_func(x, exp1[i], exp2[i], i, addition))
            return this_is_it
        
        def wrapper_fit_function(x, deaths, a, b, c, d, addition, *args):
            exp1 = args[::2]
            exp2 = args[1::2]
            return fit_function(x, a, b, c, d, addition, np.fabs(exp1), np.fabs(exp2), deaths)


        
        
        
        
        
        user = getpass.getuser()
        addition_path = '/home/' + user + '/Work/Fits/SAA_additions/' + str(day) + '/'
        if counts.all == total_rate.all:
            addition_name = detector.__name__ + '_add_tot.dat'
        else:
            addition_name = detector.__name__ + '_add_e' + str(echan) + '.dat'
        addition_file = os.path.join(addition_path, addition_name)
        
        
        
        
        
        
        
        
        
        if os.access(addition_file, os.F_OK):
            infos = open(addition_file, 'r')
            addition = int(infos.read())
            infos.close()
            
        else:
            if not os.access(addition_path, os.F_OK):
                print("Making New Directory")
                os.mkdir(addition_path)
    
            good_fit = []
            
            for addition in range(0, 20):
                
                x0 = np.append(np.array([1300., 20., -12., -1., addition]), exp)
                sigma = np.array((counts + 1)**(0.5))
                
                try:
                    fit_results = optimization.curve_fit(lambda x, a, b, c, d, addition, *args: wrapper_fit_function(x, deaths, a, b, c, d, addition, *args), bin_time_mid, counts, x0, sigma, maxfev = 10000)
                
                except RuntimeError:
                    print("Error - curve_fit failed for the value " + str(addition) + ".")
                
                coeff = fit_results[0]
                
                a = coeff[0]
                b = coeff[1]
                c = coeff[2]
                d = coeff[3]
                addition = coeff[4]
                exp1 = coeff[5::2]
                exp2 = coeff[6::2]
                
                fit_curve = fit_function(bin_time_mid, a, b, c, d, addition, exp1, exp2, deaths)
                
                residual_curve = counts - fit_curve
                
                chi_squared = np.sum(residual_curve**2)
            
                good_fit.append(chi_squared)
            
            good_fit = np.array(good_fit)
        
            print np.argmin(good_fit)
        
            addition = np.argmin(good_fit)
            
            infos = open(addition_file, 'w')
            infos.write(str(addition))
            infos.close()
        
        
        
        
        
        
        x0 = np.append(np.array([1300., 20., -12., -1., addition]), exp)
        sigma = np.array((counts + 1)**(0.5))
        
        fit_results = optimization.curve_fit(lambda x, a, b, c, d, addition, *args: wrapper_fit_function(x, deaths, a, b, c, d, addition, *args), bin_time_mid, counts, x0, sigma, maxfev = 10000)
        coeff = fit_results[0]
        #pcov = fit_results[1]
        
        #print 'pcov: ', pcov
        #print 'CGB coefficient:',coeff[0]
        #print 'Magnetic field coefficient:',coeff[1]
        #print 'Earth angle coefficient:',coeff[2]
        #print 'Sun angle coefficient:',coeff[3]
        #print 'Addition: ', coeff[4]
        #print 'starting exp:',coeff[5],'&',coeff[6]
        #print 'first SAA:',coeff[7],'&',coeff[8]
        
        
        
        
        
        
        a = coeff[0]
        b = coeff[1]
        c = coeff[2]
        d = coeff[3]
        addition = coeff[4]
        exp1 = coeff[5::2]
        exp2 = coeff[6::2]
        
        fit_curve = fit_function(bin_time_mid, a, b, c, d, addition, exp1, exp2, deaths)
        
        
        
        
        
        
        
        
        
        #####plot-algorhythm#####
        #convert the x-axis into hours of the day
        plot_time_bin_date = calculate.met_to_date(calculate(), bin_time_mid)[0]
        plot_time_bin = (plot_time_bin_date - calc.day_to_met(day)[1])*24#Time of day in hours
        plot_time_sat_date = calculate.met_to_date(calculate(), sat_time_bin)[0]
        plot_time_sat = (plot_time_sat_date - calc.day_to_met(day)[1])*24#Time of day in hours
        
        
        ###plot each on the same axis as converted to counts###
        fig, ax1 = plt.subplots()
        
        plot1 = ax1.plot(plot_time_bin, counts, 'b-', label = 'Countrate')
        plot2 = ax1.plot(plot_time_bin, fit_curve, 'r-', label = 'Fit')
        plot3 = ax1.plot(plot_time_sat, d*sun_ang_bin, 'y-', label = 'Sun angle')
        plot4 = ax1.plot(plot_time_sat, c*earth_ang_bin, 'c-', label = 'Earth angle')
        plot5 = ax1.plot(plot_time_sat, b*magnetic, 'g-', label = 'Magnetic field')
        plot6 = ax1.plot(plot_time_sat, a*cgb, 'b--', label = 'Cosmic y-ray background')
        #plot7 = ax1.plot(plot_time_sat, j2000_orb, 'y--', label = 'J2000 orbit')
        #plot8 = ax1.plot(plot_time_sat, geo_orb, 'g--', label = 'Geographical orbit')
        
        #plot vertical lines for the solar flares of the day
        if np.all(flares_today != -5):
            if len(flares_today[0]) > 1:
                for i in range(0, len(flares_today[0])):
                    plt.axvline(x = flares_today[0,i], ymin = 0., ymax = 1., linewidth=2, color = 'grey')
                    plt.axvline(x = flares_today[1,i], ymin = 0., ymax = 1., color = 'grey', linestyle = '--')
            else:
                plt.axvline(x = flares_today[0], ymin = 0., ymax = 1., linewidth=2, color = 'grey')
                plt.axvline(x = flares_today[1], ymin = 0., ymax = 1., color = 'grey', linestyle = '--')
        
        plots = plot1 + plot2 + plot3 + plot4 + plot5 + plot6
        labels = [l.get_label() for l in plots]
        ax1.legend(plots, labels, loc=1)
        
        ax1.grid()
        
        ax1.set_xlabel('Time of day in 24h')
        ax1.set_ylabel('Countrate')
        
        #ax1.set_xlim([9.84, 9.85])
        ax1.set_xlim([-0.5, 24.5])
        ax1.set_ylim([-500, 1500])
        
        plt.title(data_type + '-countrate-fit of the ' + detector.__name__ + '-detector on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(year))
        
        figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
        if not os.access(figure_path, os.F_OK):
            os.mkdir(figure_path)
        if counts.all == total_rate.all:
            figure_name = detector.__name__ + '_tot.png'
        else:
            figure_name = detector.__name__ + '_e' + str(echan) + '.png'
        
        figure = os.path.join(figure_path, figure_name)
        plt.savefig(figure, bbox_inches='tight')
        
        #plt.show()
        
        
        
        
        
        
        
        ###plot residual noise of the fitting algorithm###
        plt.plot(plot_time_bin, counts - fit_curve, 'b-')
        
        plt.xlabel('Time of day in 24h')
        plt.ylabel('Residual noise')
        
        plt.grid()
        
        plt.title(data_type + '-counts-fit residuals of the ' + detector.__name__ + '-detector on the ' + ordinal(int(str(day)[4:6])) + ' ' + date.strftime('%B')[0:3] + ' ' + str(year))
        
        plt.ylim([-600, 600])
        
        figure_path = '/home/' + user + '/Work/Fits/' + str(day) + '/'
        if not os.access(figure_path, os.F_OK):
            os.mkdir(figure_path)
        if counts.all == total_rate.all:
            figure_name = detector.__name__ + '_tot_res.png'
        else:
            figure_name = detector.__name__ + '_e' + str(echan) + '_res.png'
        
        figure = os.path.join(figure_path, figure_name)
        plt.savefig(figure,bbox_inches="tight")
        
        #plt.show()
        

        
        return 