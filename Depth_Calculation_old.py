# -*- coding: utf-8 -*-
"""
hydrogen depth measurements

Created on Tue Mar 25 12:38:06 2014

@author: Christoph Mueller
<christoph-1.mueller@uni-ulm.de>
"""

import matplotlib
import numpy as np
import pylab
import os


from enthought.traits.api import *
from enthought.traits.ui.api import *
from enthought.enable.api import Component, ComponentEditor
from enthought.chaco.api import ArrayDataSource, LinePlot, LinearMapper, ArrayPlotData, Plot, CMapImagePlot
from enthought.chaco.tools.api import PanTool, ZoomTool
from enthought.chaco.tools.cursor_tool import CursorTool, CursorTool1D

import Fit
import Calculate


class Depth_Calculation(HasTraits):
    
    xy8_file = File(label = 'XY8-file:')
    rabi_file = File(label = 'rabi-file:')
    
    gamma = 2*np.pi*2.8025*1e4  #in MHz/T
    gamma_e = Float(gamma, label = 'gamma_e (MHz/T)')
    Number_pulses = Int(0, label = 'Number of pulses:')
    Number_xy8_rep = Int(0, label = 'Number of xy8-repetitions:')
    proton_density = Float(50, label = 'proton density (1/nm^3)') 
    spec_min = Float(1, label = 'spec min (MHz)')
    spec_max = Float(1, label = 'spec max (MHz)')
    
    Calculate_spectrum = Button(label = 'Calculate spectrum')
    Calculate_depth = Button(label = 'Calculate depth')
    depth1 = Float(0.)
    depth2 = Float(0.)
    save_results = Button(label = 'save results')
    
    linestoignore = 1
    nu = []
    lorentz_fit_parameters_y = []
    lorentz_fit_parameters_yy = []
    
    rabi_plot_x = Array()
    rabi_plot_y = Array()
    rabi_plot_fit_y = Array()
    density_plot_x = Array()
    density_plot_y = Array()
    density_plot_yy = Array()
    density_plot_fit_y = Array()
    density_plot_fit_yy = Array()
    xy8_plot_x = Array()
    xy8_plot_y = Array()
    xy8_plot_yy = Array()
    xy8norm_plot_x = Array()
    xy8norm_plot_y = Array()
    xy8norm_plot_yy = Array()
    depth_plot_x = Array()
    depth_plot_y_low = Array()
    depth_plot_y_high = Array()
    
    plot_rabi = Instance(Plot, transient=True )
    plot_rabi_data = Instance( ArrayPlotData, transient=True )
    plot_density = Instance(Plot, transient=True )
    plot_density_data = Instance( ArrayPlotData, transient=True )
    plot_xy8 = Instance(Plot, transient=True )
    plot_xy8_data = Instance( ArrayPlotData, transient=True )
    plot_xy8norm = Instance(Plot, transient=True )
    plot_xy8norm_data = Instance( ArrayPlotData, transient=True )
    colormap = ['blue', 'red', 'green', 'orange']
    
    
    
############################  GUI   ##################################   
    traits_view = View(
          VGroup(Item('rabi_file',label='rabi file', editor=FileEditor(filter=['*.asc'])),
                 Item('xy8_file',label='xy8 file', editor=FileEditor(filter=['*.asc'])),
                 HGroup(Item('gamma_e', show_label=True),
                        Item('proton_density', show_label=True),
                        Item('Number_xy8_rep', show_label=True),
                        Item('Number_pulses', show_label=True)
                        ),
                 HGroup(Item('spec_min', show_label=True),
                        Item('spec_max', show_label=True),
                        Item('Calculate_spectrum', show_label=False),
                        Item('Calculate_depth', show_label=False)
                        ),
                 HGroup(VGroup(
                        HGroup(Item('plot_rabi', editor=ComponentEditor(), show_label=False, width=500, height=350, resizable=True),
                               Item('plot_xy8norm', editor=ComponentEditor(), show_label=False, width=500, height=350, resizable=True)
                               ),
                        HGroup(Item('plot_xy8', editor=ComponentEditor(), show_label=False, width=500, height=350, resizable=True),
                               Item('plot_density', editor=ComponentEditor(), show_label=False, width=500, height=350, resizable=True)
                               )
                               ),
                        VGroup(Item(label='          #####    results:    #####          '),
                               HGroup(Item('depth1', style='readonly')),
                               HGroup(Item('depth2', style='readonly')),
                               Item('save_results', show_label=False)
                               )
                        )
                 ))
######################################################################


    def _Calculate_spectrum_fired(self):
        # delete values from previous measurement
        self.delete_all_values()
        
        # load data
        rabi_x, rabi_y, unused1, unused2 = self.load_file(self.rabi_file)
        xy8_x, xy8_y, xy8_xx, xy8_yy = self.load_file(self.xy8_file)
        # performing rabi-fit
        rabi_fit_parameters = self.rabi_fit(rabi_x, rabi_y)
        rabi_contrast = 200*rabi_fit_parameters[0]/rabi_fit_parameters[3]
        rabi_period = rabi_fit_parameters[1]/1e3  # /1e3 gives us nanoseconds
        rabi_amplitude = rabi_fit_parameters[0]
        rabi_level = rabi_fit_parameters[3]
        print 'rabi_level: ', rabi_level
        # plotting rabi + rabi-fit
        rabi_y = np.array(rabi_y)/1e3
        self.y_rabi_lo = min(rabi_y) - 0.02
        self.y_rabi_hi = max(rabi_y) + 0.02
        rabi_fit_y = Fit.Cosinus_dec(*rabi_fit_parameters)(rabi_x)/1e3
        self._plot_rabi_changed(rabi_x, rabi_y, rabi_fit_y)
        # plotting yx8-data
        self._plot_xy8_changed(xy8_x, xy8_y, xy8_yy)
        # decide if rabi-level or xy8-level (if alternating)
        level = rabi_level
        if xy8_yy != []:
            xy8_level = ((np.array(xy8_y)+np.array(xy8_yy))/2).mean()
            level = xy8_level
            print 'xy8_level: ', xy8_level
        # performing xy8 to rabi normalization        
        xy8norm_x = xy8_x
        xy8norm_y = Calculate.do_normalization(xy8_y, rabi_amplitude, level)
        xy8norm_yy = []
        if xy8_yy != []:
            xy8norm_yy = Calculate.do_normalization(xy8_yy, rabi_amplitude, level)
        # plotting xy8norm
        self._plot_xy8norm_changed(xy8norm_x, xy8norm_y, xy8norm_yy)
        # calculating the spectral density
        nu = Calculate.nu_from_tau(xy8norm_x) #this nu is in MHz, since xy8norm_x is in µs
        gamma_e = 1 #setting gamma_e to 1
        tau_x = np.array(xy8norm_x)/1e6 #tau_x is in seconds
        if xy8_yy == []:
            temp = float(sum(xy8norm_y)/len(xy8norm_y))
            if temp < 0.5:
                xy8norm_y = 1 - xy8norm_y                
        spectrum_y = Calculate.S_from_data(tau_x, xy8norm_y, gamma_e, self.Number_pulses)
        spectrum_y = np.array(spectrum_y)/1e6 #to change to MHz
        spectrum_yy = []
        if xy8_yy != []:
            xy8norm_yy = 1 - xy8norm_yy
            spectrum_yy = Calculate.S_from_data(tau_x, xy8norm_yy, gamma_e, self.Number_pulses)
            spectrum_yy = np.array(spectrum_yy)/1e6 #to change to MHz        
        # plotting the spectral density
        self.spec_min = min(nu)
        self.spec_max = max(nu)
        self._plot_density_changed(nu, spectrum_y, spectrum_yy)
        # saving nu and spectra, to use them in Calculate_depth
        self.nu = nu
        self.spectrum_y = spectrum_y
        self.spectrum_yy = spectrum_yy


    def _Calculate_depth_fired(self):
        # take the values from the calculated spectral density
        nu = np.array(self.nu)
        spectrum_y = np.array(self.spectrum_y)
        spectrum_yy = np.array(self.spectrum_yy)
        
        nu_calc = nu[np.logical_and(nu>=self.spec_min, nu<=self.spec_max)]
        spectrum_y_calc = []        
        spectrum_yy_calc = []
        i=0
        for i in range(0,len(nu)):
            if nu[i] in nu_calc:
                spectrum_y_calc.append(spectrum_y[i])
                if spectrum_yy != []:                    
                    spectrum_yy_calc.append(spectrum_yy[i])
        # performing lorentzian fits to the spectral data
        spectrum_y_calc = np.array(spectrum_y_calc)
        self.lorentz_fit_parameters_y = Fit.Fit(nu_calc, spectrum_y_calc, Fit.Lorentzian, Fit.LorentzianEstimator)
        self.density_plot_fit_y = Fit.Lorentzian(*self.lorentz_fit_parameters_y)(nu)        
        if spectrum_yy != []:
            spectrum_yy_calc = np.array(spectrum_yy_calc)            
            self.lorentz_fit_parameters_yy = Fit.Fit(nu_calc, spectrum_yy_calc, Fit.Lorentzian, Fit.LorentzianEstimator)
            self.density_plot_fit_yy = Fit.Lorentzian(*self.lorentz_fit_parameters_yy)(nu)
        # update the spectral density plots with the fits in it
        self._plot_density_changed(nu, spectrum_y, spectrum_yy)
        # calculate the area from the lorentz fit
        gamma_e = self.gamma_e
        rho = self.proton_density*1e27 #to get 1/m^3
        Area1 = self.lorentz_fit_parameters_y[2]#in MHz^2
        print Area1
        #Area1 = 0.0721
        B1 = abs(2*Area1/(gamma_e*gamma_e))**(0.5) #in Tesla
        self.depth1 = Calculate.calculate_depth_simple(B1, rho)
        print B1
                
        if spectrum_yy != []:
            Area2 = self.lorentz_fit_parameters_yy[2]#in MHz^2
            print Area2
            B2 = abs(2*Area2/(gamma_e*gamma_e))**(0.5) #in Tesla
            self.depth2 = Calculate.calculate_depth_simple(B2, rho)
            print B2




    def _spec_min_changed(self):        
        self._plot_density_changed(self.nu, self.spectrum_y, self.spectrum_yy)
        
    def _spec_max_changed(self):        
        self._plot_density_changed(self.nu, self.spectrum_y, self.spectrum_yy)
        
    def delete_all_values(self):
        self.spectrum_y = [1.0,2.0]
        self.spectrum_yy = [1.0,2.0]
        self.density_plot_yy = []
        self.density_plot_fit_y = []
        self.density_plot_fit_yy = []
        self.xy8_plot_yy = []
        self.xy8norm_plot_yy = []
        self.depth1 = 0.
        self.depth2 = 0.
    
    
    def _xy8_file_changed(self):
#        temp = self.xy8_file
#        i = -11
#        value = 0
#        while temp[i:-10].isalnum() == True:
#            value = int(temp[i:-10])
#            i = i-1
#        self.Number_xy8_rep = value
        pass
        

    def _Number_xy8_rep_changed(self):
        temp = self.Number_xy8_rep
        value = temp*8
        self.Number_pulses = value
                

    def load_file(self, filepath):
        """ loads the data from a file that has a 2 column structure
            ignores the first linestoignore lines of this file """
        x, y, xx, yy = self.read_data(filepath)
        return x, y, xx, yy
        
    
    def read_data(self, filepath):
        """importing data from a file"""
        isAlternating = False
        x = []
        y = []
        xx = []
        yy = []
        inp = open (filepath,'rb')
        check = inp.readline()  
        check_new = check.split()
        if len(check_new) > 5:
            isAlternating = True            
        inp.seek(0)       #moves the cursor back to the beginning of the file
        i = 0
        for line in inp:
            i += 1
            if i > self.linestoignore:
                numbers = map(float, line.split())
                x.append(numbers[0])          # now x is in microseconds
                y.append(numbers[1]*1e3)                    
                if isAlternating:
                    if check_new[3] == 'Fit':
                        xx.append(numbers[3])     # now xx is in microseconds
                        yy.append(numbers[4]*1e3)                        
                    else:
                        xx.append(numbers[2])     # now yy is in microseconds
                        yy.append(numbers[3]*1e3)
        inp.close()
        #checken, ob  len(x) = len(xx)
        return x, y, xx, yy
        

    def rabi_fit(self, rabi_x, rabi_y):
        x = rabi_x
        y = rabi_y
        x = np.array(x)
        y = np.array(y)
        y_offset = y.mean()
        yreal = y - y_offset
        try:
            fit_parameters = Fit.Fit(x, yreal, Fit.CosinusNoOffset, Fit.CosinusNoOffsetEstimator)
        except:
            print 'Error'
            return None
        if fit_parameters[0] < 0:
            fit_parameters[0] = -fit_parameters[0]
            fit_parameters[2] =  ( ( fit_parameters[2]/fit_parameters[1] + 0.5 ) % 1 ) * fit_parameters[1]
            fit_parameters = Fit.Fit(x, yreal, Fit.CosinusNoOffset, fit_parameters)
        fit_parameters = (fit_parameters[0], fit_parameters[1], fit_parameters[2], y_offset)
        fit_parameters = Fit.Fit(x, y, Fit.Cosinus, fit_parameters)
        while(fit_parameters[2]>0.5*fit_parameters[1]):
            fit_parameters[2] -= fit_parameters[1]
        fit_parameters = Fit.Fit(x, y, Fit.Cosinus, fit_parameters)
        fit_parameters = list(fit_parameters)
        fit_parameters.append(10*max(x))
        fit_parameters = Fit.Fit(x, y, Fit.Cosinus_dec, fit_parameters)
        return fit_parameters
        


    def _plot_rabi_changed(self, rabi_x, rabi_y, rabi_fit_y):        
        self.rabi_plot_x = rabi_x
        self.rabi_plot_y = rabi_y
        self.rabi_plot_fit_y = rabi_fit_y
        self.plot_rabi.y_axis.mapper.range.set(low=self.y_rabi_lo, high=self.y_rabi_hi)
        self.plot_rabi_data.set_data('rabi_plot_x', self.rabi_plot_x)
        self.plot_rabi_data.set_data('rabi_plot_y', self.rabi_plot_y)  
        self.plot_rabi_data.set_data('rabi_plot_x', self.rabi_plot_x)
        self.plot_rabi_data.set_data('rabi_plot_fit_y', self.rabi_plot_fit_y)
        self.plot_rabi.plot(('rabi_plot_x','rabi_plot_fit_y'), style='line', color='red')
        self.plot_rabi.request_redraw()
        
    def _plot_xy8_changed(self, xy8_x, xy8_y, xy8_yy):        
        self.xy8_plot_x = xy8_x
        self.xy8_plot_y = np.array(xy8_y)/1e3
        self.xy8_plot_yy = np.array(xy8_yy)/1e3
        self.plot_xy8.y_axis.mapper.range.set(low=self.y_rabi_lo, high=self.y_rabi_hi)
        self.plot_xy8_data.set_data('xy8_plot_x', self.xy8_plot_x)
        self.plot_xy8_data.set_data('xy8_plot_y', self.xy8_plot_y)  
        #if xy8_yy != []:
        self.plot_xy8_data.set_data('xy8_plot_x', self.xy8_plot_x)
        self.plot_xy8_data.set_data('xy8_plot_yy', self.xy8_plot_yy)
        self.plot_xy8.plot(('xy8_plot_x','xy8_plot_yy'), style='line', color='green')
        self.plot_xy8.request_redraw()   
        
    def _plot_xy8norm_changed(self, xy8norm_x, xy8norm_y, xy8norm_yy):        
        self.xy8norm_plot_x = xy8norm_x
        self.xy8norm_plot_y = xy8norm_y
        self.xy8norm_plot_yy = xy8norm_yy
        self.plot_xy8norm_data.set_data('xy8norm_plot_x', self.xy8norm_plot_x)
        self.plot_xy8norm_data.set_data('xy8norm_plot_y', self.xy8norm_plot_y)  
        #if xy8norm_yy != []:
        self.plot_xy8norm_data.set_data('xy8norm_plot_x', self.xy8norm_plot_x)
        self.plot_xy8norm_data.set_data('xy8norm_plot_yy', self.xy8norm_plot_yy)
        self.plot_xy8norm.plot(('xy8norm_plot_x','xy8norm_plot_yy'), style='line', color='green')
        self.plot_xy8norm.request_redraw()    
        
    def _plot_density_changed(self, nu, density_y, density_yy):        
        self.density_plot_x = nu
        self.density_plot_y = density_y
        self.density_plot_yy = density_yy
        self.plot_density.x_axis.mapper.range.set(low=self.spec_min, high=self.spec_max)
        self.plot_density_data.set_data('density_plot_x', self.density_plot_x)
        self.plot_density_data.set_data('density_plot_y', self.density_plot_y)  
        #if density_yy != []:
        self.plot_density_data.set_data('density_plot_x', self.density_plot_x)
        self.plot_density_data.set_data('density_plot_yy', self.density_plot_yy)
        self.plot_density.plot(('density_plot_x','density_plot_yy'), style='line', color='green')
        if self.lorentz_fit_parameters_y != []:
            self.plot_density_data.set_data('density_plot_x', self.density_plot_x)
            self.plot_density_data.set_data('density_plot_fit_y', self.density_plot_fit_y)
            self.plot_density.plot(('density_plot_x','density_plot_fit_y'), style='line', color='red')
        if self.lorentz_fit_parameters_yy != []:
            self.plot_density_data.set_data('density_plot_x', self.density_plot_x)
            self.plot_density_data.set_data('density_plot_fit_yy', self.density_plot_fit_yy)
            self.plot_density.plot(('density_plot_x','density_plot_fit_yy'), style='line', color='orange')
        self.plot_density.request_redraw()     
        
######## plot default values ##########

    def _plot_rabi_data_default(self):
        return ArrayPlotData(rabi_plot_x=self.rabi_plot_x, rabi_plot_y=self.rabi_plot_y)

    def _plot_rabi_default(self):
        plot = Plot(self.plot_rabi_data, padding_left=60, padding_top=25, padding_right=10, padding_bottom=50)
        plot.plot(('rabi_plot_x','rabi_plot_y'), style='line', color='blue')
        plot.index_axis.title = 'tau [micro-s]'
        plot.value_axis.title = 'Intensity [a.u.]'
        plot.title = 'Rabi raw data'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        return plot

    def _plot_density_data_default(self):
        return ArrayPlotData(density_plot_x=self.density_plot_x, density_plot_y=self.density_plot_y)

    def _plot_density_default(self):
        plot = Plot(self.plot_density_data, padding_left=60, padding_top=25, padding_right=10, padding_bottom=50)
        plot.plot(('density_plot_x','density_plot_y'), style='line', color='blue')
        plot.index_axis.title = 'frequency [MHz]'
        plot.value_axis.title = 'spectral density [MHz]'
        plot.title = 'Spectral density'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        return plot
        
    def _plot_xy8_data_default(self):
        return ArrayPlotData(xy8_plot_x=self.xy8_plot_x, xy8_plot_y=self.xy8_plot_y)

    def _plot_xy8_default(self):
        plot = Plot(self.plot_xy8_data, padding_left=60, padding_top=25, padding_right=10, padding_bottom=50)
        plot.plot(('xy8_plot_x','xy8_plot_y'), style='line', color='blue')
        plot.index_axis.title = 'tau [micro-s]'
        plot.value_axis.title = 'Intensity [a.u.]'
        plot.title = 'xy8 raw data'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        return plot
        
    def _plot_xy8norm_data_default(self):
        return ArrayPlotData(xy8norm_plot_x=self.xy8norm_plot_x, xy8norm_plot_y=self.xy8norm_plot_y)

    def _plot_xy8norm_default(self):
        plot = Plot(self.plot_xy8norm_data, padding_left=60, padding_top=25, padding_right=10, padding_bottom=50)
        plot.plot(('xy8norm_plot_x','xy8norm_plot_y'), style='line', color='blue')
        plot.index_axis.title = 'tau [micro-s]'
        plot.value_axis.title = 'normalized Intensity'
        plot.title = 'xy8 normalized to rabi contrast'
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        plot.y_axis.mapper.range.set(low=-0.1, high=1.1)
        return plot
        
        
#################### saving the data  ###############################

    def _save_results_changed(self):
        self.foldername_tmp = self.rabi_file[:-9] + '_depth_meas_results'
        self.foldername = self.foldername_tmp
        i = 0
        while True:
            try:
                os.mkdir(self.foldername)
                break
            except:
                i += 1
                self.foldername = self.foldername_tmp + '(' + str(i) + ')'
                
        self._save_asc('rabi', self.rabi_plot_x, self.rabi_plot_y, self.rabi_plot_fit_y)
        self._save_asc('xy8raw', self.xy8_plot_x, self.xy8_plot_y, self.xy8_plot_yy)
        self._save_asc('xy8norm', self.xy8norm_plot_x, self.xy8norm_plot_y, self.xy8norm_plot_yy)
        self._save_asc('spec', self.density_plot_x, self.density_plot_y, self.density_plot_yy, self.density_plot_fit_y, self.density_plot_fit_yy)
        
        self._save_png('rabi', self.rabi_plot_x, self.rabi_plot_y, self.rabi_plot_fit_y)
        self._save_png('xy8raw', self.xy8_plot_x, self.xy8_plot_y, self.xy8_plot_yy)
        self._save_png('xy8norm', self.xy8norm_plot_x, self.xy8norm_plot_y, self.xy8norm_plot_yy)
        self._save_png('spec', self.density_plot_x, self.density_plot_y, self.density_plot_yy, self.density_plot_fit_y, self.density_plot_fit_yy)
        
        
        new_file = self.foldername + '\\' + 'depths.asc'
        fil = open(new_file, 'w')
        fil.write('depth1 = %4.5f nm\n'%self.depth1)
        fil.write('depth2 = %4.5f nm\n'%self.depth2)
        fil.close()
        
        
    def _save_asc(self, filename, x=[], y1=[], y2=[], y3=[], y4=[]):
        new_file = self.foldername + '\\' + filename + '_results.asc'
        fil = open(new_file, 'w')
        if filename == 'rabi':
            fil.write('tau[micro-s] data fit\n')
            for i in range(0,len(x)):
                fil.write('%4.3f %4.3f %4.3f\n'%(x[i],y1[i],y2[i]))
            fil.close()
        if filename == 'xy8raw' or filename == 'xy8norm':
            if y2 != []:
                fil.write('tau[micro-s] data1 data2\n')
                for i in range(0,len(x)):
                    fil.write('%4.3f %4.3f %4.3f\n'%(x[i],y1[i],y2[i]))
                fil.close()
            else:
                fil.write('tau[micro-s] data\n')
                for i in range(0,len(x)):
                    fil.write('%4.3f %4.3f\n'%(x[i],y1[i]))
                fil.close()
        if filename == 'spec':
            if y2 != []:
                fil.write('frequency[MHz] data1 data2 fit1 fit2\n')
                for i in range(0,len(x)):
                    fil.write('%4.3f %4.3f %4.3f %4.5f %4.5f\n'%(x[i],y1[i],y2[i],y3[i],y4[i]))
                fil.close()
            else:
                fil.write('frequency[MHz] data1 fit\n')
                for i in range(0,len(x)):
                    fil.write('%4.3f %4.3f %4.5f\n'%(x[i],y1[i],y3[i]))
                fil.close()
                
    
    def _save_png(self, filename, x=[], y1=[], y2=[], y3=[], y4=[]):
        new_file = self.foldername + '\\' + filename + '_results.png'
        pylab.rcParams['figure.figsize'] = 8, 5
        if filename == 'rabi':        
            pylab.plot(x, y1, 'b-', linewidth=1.5, label='data')
            pylab.plot(x, y2, 'r-', linewidth=1.5, label='fit')
            pylab.xlabel('tau [micro-s]')
            pylab.ylabel('Intensity [a.u.]')
            pylab.title('Rabi raw data')
            pylab.minorticks_on()
            pylab.savefig(new_file)
            pylab.close()
        if filename == 'xy8raw':
            pylab.plot(x, y1, 'b-', linewidth=1.5, label='data1')    
            if y2 !=[]:
                pylab.plot(x, y2, 'g-', linewidth=1.5, label='data2')
            pylab.xlabel('tau [micro-s]')
            pylab.ylabel('Intensity [a.u.]')
            pylab.title('xy8 raw data')
            pylab.minorticks_on()
            pylab.savefig(new_file)
            pylab.close()
        if filename == 'xy8norm':        
            pylab.plot(x, y1, 'b-', linewidth=1.5, label='data1')   
            if y2 !=[]:
                pylab.plot(x, y2, 'g-', linewidth=1.5, label='data2')
            pylab.xlabel('tau [micro-s]')
            pylab.ylabel('Intensity [a.u.]')
            pylab.title('xy8 norm data')
            pylab.minorticks_on()
            pylab.savefig(new_file)
            pylab.close()
        if filename == 'spec':        
            pylab.plot(x, y1, 'b-', linewidth=1.5, label='data1')   
            if y2 !=[]:
                pylab.plot(x, y2, 'g-', linewidth=1.5, label='data2')
            pylab.plot(x, y3, 'r-', linewidth=1.5, label='fit1')   
            if y2 !=[]:
                pylab.plot(x, y4, 'y-', linewidth=1.5, label='fit2')
            pylab.xlabel('frequency [MHz]')
            pylab.ylabel('spectral density [MHz]')
            pylab.title('spectral density')
            pylab.minorticks_on()
            pylab.savefig(new_file)
            pylab.close()
        
        
        
Depth_Calculation().configure_traits() 