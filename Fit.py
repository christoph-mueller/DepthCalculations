"""
This file is part of Diamond. Diamond is a confocal scanner written
in python / Qt4. It combines an intuitive gui with flexible
hardware abstraction classes.

Diamond is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Diamond is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with diamond. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2009 Helmut Rathgen <helmut.rathgen@gmail.com>
"""

import numpy
import scipy.optimize, scipy.stats
import math

def Cosinus(a, T, x0, c):
    """Returns a Cosinus function with the given parameters"""
    return lambda x: a*numpy.cos( 2*numpy.pi*(x-x0)/float(T) ) + c
setattr(Cosinus, 'Formula', r'$cos(c,a,T,x0;x)=a\cos(2\pi(x-x0)/T)+c$')

def CosinusEstimator(x, y):
    c = y.mean()
    a = 2**0.5 * numpy.sqrt( ((y-c)**2).sum() )
    # better to do estimation of period from
    Y = numpy.fft.fft(y)
    N = len(Y)
    D = float(x[1] - x[0])
    i = abs(Y[1:N/2+1]).argmax()+1
    T = (N * D) / i
    x0 = 0
    return a, T, x0, c

def Cosinus_dec(a, T, x0, c, t2):
    return lambda x: a*numpy.cos( 2*numpy.pi*(x-x0)/float(T) ) * numpy.exp(-(x-x0)/t2) + c

def Cosinus_dec_estimator(x, y):
    a, T, x0, c = CosinusEstimator(x, y)
    t2 = 10 * max(x)
    return a, T, x0, c, t2

def Cosinus_dec_nophase(a, T, c, t2):
    return lambda x: a*numpy.cos( 2*numpy.pi*(x)/float(T) ) * numpy.exp(-(x)/t2) + c

def Cosinus_dec_nophase_estimator(x, y):
    a, T, x0, c = CosinusEstimator(x, y)
    t2 = 10 * max(x)
    return a, T, c, t2

def cosinus_nophase(amp, per, c):
    """Returns a Cosinus function with phase = 0."""
    return lambda x: amp * numpy.cos(2*numpy.pi*numpy.array(x)/float(per)) + c

def cosinus_nophase_estimator(x, y):
    x = numpy.array(x)
    y = numpy.array(y)
    c = y.mean()
    a = y.max() - y.min()
    if y[0] <= c: a = -a
    # better to do estimation of period from
    Y = numpy.fft.fft(y)
    N = len(Y)
    D = float(x[1] - x[0])
    i = abs(Y[1:N/2+1]).argmax()+1
    T = (N * D) / i
    return a, T, c

def CosinusNoOffset(a, T, x0):
    """Returns a Cosinus function with the given parameters"""
    return lambda x: a*numpy.cos( 2*numpy.pi*(x-x0)/float(T) )
setattr(Cosinus, 'Formula', r'$cos(a,T,x0;x)=a\cos(2\pi(x-x0)/T)$')

def CosinusNoOffsetEstimator(x, y):
    a = 2**0.5 * numpy.sqrt( (y**2).sum() )
    # better to do estimation of period from
    Y = numpy.fft.fft(y)
    N = len(Y)
    D = float(x[1] - x[0])
    i = abs(Y[1:N/2+1]).argmax()+1
    T = (N * D) / i
    x0 = 0
    return a, T, x0

def brot_transitions_upper(B, D, E, phase):
    return lambda theta: 3./2. * B**2/D * numpy.sin(theta + phase)**2 + ( B**2 * numpy.cos(theta + phase)**2 + (E + B**2/(2*D) * numpy.sin(theta+phase)**2)**2)**0.5 + D
    
def brot_transitions_lower(B, D, E, phase):
    return lambda theta: 3./2. * B**2/D * numpy.sin(theta + phase)**2 - ( B**2 * numpy.cos(theta + phase)**2 + (E + B**2/(2*D) * numpy.sin(theta+phase)**2)**2)**0.5 + D

def FCSTranslationRotation(alpha, tau_r, tau_t, N):
    """Fluorescence Correlation Spectroscopy. g(2) accounting for translational and rotational diffusion."""
    return lambda t: (1 + alpha*numpy.exp(-t/tau_r) ) / (N * (1 + t/tau_t) )
setattr(FCSTranslationRotation, 'Formula', r'$g(\alpha,\tau_R,\tau_T,N;t)=\frac{1 + \alpha \exp(-t/\tau_R)}{N (1 + t/\tau_T)}$')

def FCSTranslation(tau, N):
    """Fluorescence Correlation Spectroscopy. g(2) accounting for translational diffusion."""
    return lambda t: 1. / (N * (1 + t/tau) )
setattr(FCSTranslation, 'Formula', r'$g(\tau,N;t)=\frac{1}{N (1 + t/\tau)}$')

def Antibunching(alpha, c, tau, t0):
    """Antibunching. g(2) accounting for Poissonian background."""
    return lambda t: c*(1-alpha*numpy.exp(-(t-t0)/tau))
setattr(Antibunching, 'Formula', r'$g(\alpha,c,\tau,t_0;t)=c(1 - \alpha \exp(-(t-t_0)/\tau))$')

def StrExp(c, a, w):
    """Gaussian function with offset."""
    return lambda x: c + a*numpy.exp(- (x/w)**2   )
setattr(StrExp, 'Formula', r'$f(c,a,w;x)=c+a\exp(-(x/w)^2)$')

def Exp(c, a, w):
    """Gaussian function with offset."""
    return lambda x: c + a*numpy.exp(- (x/w))
setattr(Exp, 'Formula', r'$f(c,a,w;x)=c+a\exp(-(x/w))$')

def Gaussian(c, a, x0, w):
    """Gaussian function with offset."""
    return lambda x: c + a*numpy.exp( -0.5*((x-x0)/w)**2   )
setattr(Gaussian, 'Formula', r'$f(c,a,x0,w;x)=c+a\exp(-0.5((x-x_0)/w)^2)$')

def DoubleGaussian(a1, a2, x01, x02, w1, w2):
    """Gaussian function with offset."""
    return lambda x: a1*numpy.exp( -0.5*((x-x01)/w1)**2   ) + a2*numpy.exp( -0.5*((x-x02)/w2)**2   )
setattr(Gaussian, 'Formula', r'$f(c,a1, a2,x01, x02,w1,w2;x)=a_1\exp(-0.5((x-x_{01})/w_1)^2)+a_2\exp(-0.5((x-x_{02})/w_2)^2)$')

def DoubleGaussianEstimator(x, y):
	center = (x*y).sum() / y.sum()
	ylow = y[x < center]
	yhigh = y[x > center]
	x01 = x[ylow.argmax()]
	x02 = x[len(ylow)+yhigh.argmax()]
	a1 = ylow.max()
	a2 = yhigh.max()
	w1 = w2 = center**0.5
	return a1, a2, x01, x02, w1, w2

# important note: lorentzian can also be parametrized with an a' instead of a,
# such that a' is directly related to the amplitude (a'=f(x=x0)). In this case a'=a/(pi*g)
# and f = a * g**2 / ( (x-x0)**2 + g**2 ) + c.
# However, this results in much poorer fitting success. Probably the g**2 in the numerator
# causes problems in Levenberg-Marquardt algorithm when derivatives
# w.r.t the parameters are evaluated. Therefore it is strongly recommended
# to stick to the parametrization given below.

#erstellen einer Minimasuchen, grob, 3 moegliche Faelle
def MinimasearchODMRcrude(x,y):
    #print x
    #print y
    b=(len(x)/20)#ergibt wieder int(wichtig!), d.h. ab 5% (da in beide richtungen der wert genommen wird, muss man nur 2.5% nehmen) abweichung vom mittwelwert darf man davon ausgehen, dass es zwei pieks gibt
    Doublecurve=bool  #wichtig damit man aufgrund der Ausgabe der "Funktion" weis, was alsnaechstes in der Aufrufenden Funktion passieren soll
    
    D=2.87e9 # wenn gegebene x-werte mit 1e3 multipliziert werden
    #suchen des xWertes am naechsten an D dran und dessen Index
    a=[]
    i=0
    while(x[i]<D):
        i=i+1    
    if(x[i+1]<=D):
        i=i+1
    #i gibt nun den Index an, um den die Fallunterscheidung gemacht wird 
    y1=y[:i]
    y2=y[i:]
    a1=y1.argmin()
    a2=y2.argmin()
    y11=y[:i-b]
    y22=y[i+b:]
    a11=y11.argmin()
    a22=y22.argmin()    
    #Fall eins: zwei weit voneinander entfernet Minima beidseitig von D    
    if (a11==a1 or a22==a2):
        Doublecurve=True
        #a2+i is necessary as it has to be the position in the whole string, not just in the part of the string
        return Doublecurve, x[a1],x[a2+i],a1,a2+i#ausgabe der beiden anfangsparameter der minima
    
    
    #Fall von beide Minima rechts von Linie
    if(min(y1)>(min(y2)+max(y))/2):
      #selber Trick wie vorher, nur gibt es jetzt garantiert 2 maxima und man muss nur das 2 kleinste finden, dass nicht ganz direkt nebenm dem ersten liegt
      b=(len(x)/40)#nochmal ueberlegen,. ob nicht vielleicht sogar nur 1% als ab stand ausreichen wuerdem, anstatt 2,5%...
      y11=y2[:a2-b]
      y22=y2[a2+b:]
      a11=y11.argmin()
      a12=y22.argmin()
      if(y2[a11]>y2[a12+b]):#schauen welches das 'kleinere' Minimum ist
          #change of minima position
          a1=a2
          a2=a12+b
      else:
          a1=a11
      Doublecurve=True
      return  Doublecurve, x[a1+i], x[a2+i],a1+i,a2+i
    else:#dann ist es nur eine Lorentzkurve, wichtiger Fall, dann kann man das magnetfeld gleich naeher ran bringen
       a=y.argmin()#muss man sich noch ueberlegen ob sich fitten dann ueberhaupt lohnt, oder nicht gleich ein anderer Schritt in die Wege geleitet werden sollte
       Doublecurve=False
       return Doublecurve, x[a],x[a],a,a
       

def Lorentzian(x0, g, a, c):
    """Lorentzian centered at x0, with amplitude (probably Area) a, offset y0 and HWHM g."""
    return lambda x: a / numpy.pi * (  g / ( (x-x0)**2 + g**2 )  ) + c  # I think a is the Area here...
setattr(Lorentzian, 'Formula', r'$f(x0,g,a,c;x)=a/\pi (g/((x-x_0)^2+g^2)) + c$')

def LorentzianEstimator(x, y):
    c = scipy.stats.mode(y)[0][0]
    yp = y - c
    Y = numpy.sum(yp) * (x[-1] - x[0]) / len(x)
    ymin = yp.min()
    ymax = yp.max()
    if ymax > abs(ymin):
        y0 = ymax
    else:
        y0 = ymin
    x0 = x[y.argmax()]
    g = Y / (numpy.pi * y0)
    a = y0 * numpy.pi * g
    return x0, g, a, c

def Lorentzian_neg(x0, g, a, c):
    """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
    return lambda x: -abs(a) / numpy.pi * (  abs(g) / ( (x-x0)**2 + g**2 )  ) + c
setattr(Lorentzian, 'Formula', r'$f(x0,g,a,c;x)=a/\pi (g/((x-x_0)^2+g^2)) + c$')

def LorentzianEstimator_neg(x, y):
    #c = scipy.stats.mode(y)[0][0]
    c = scipy.mean(y)
    yp = numpy.array(y - c)
    c = scipy.mean(c+abs(yp))
    yp = y - c
    Y = numpy.sum(yp) * (x[-1] - x[0]) / len(x)
    y0 = yp.min()
    x0 = x[y.argmin()]
    g = Y / (numpy.pi * y0)
    a = y0 * numpy.pi * g
    return x0, g, a, c

def Lorentzian_pos(x0, g, a, c):
    """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
    return lambda x: abs(a) / numpy.pi * (  g / ( (x-x0)**2 + g**2 )  ) + c
setattr(Lorentzian, 'Formula', r'$f(x0,g,a,c;x)=a/\pi (g/((x-x_0)^2+g^2)) + c$')

def LorentzianEstimator_pos(x, y):
    """x0 = shift, g = ?, a = amplitude, c = offset """
    c = scipy.stats.mode(y)[0][0]
    yp = y - c
    Y = numpy.sum(yp) * (x[-1] - x[0]) / len(x)
    y0 = yp.max()
    #ymin = yp.min()
    #ymax = yp.max()
    #if ymax > abs(ymin):
    #    y0 = ymax
    #else:
    #    y0 = ymin
    x0 = x[y.argmax()]
    g = Y / (numpy.pi * y0)
    a = y0 * numpy.pi * g
    return x0, g, a, c



def DoubleLorentzianEstimator_neg(x,y):
    Dec=True
    Dec, x02,x03,a02,a03 =MinimasearchODMRcrude(x,y)
    #define break between minima
    a=(a03+a02)/2
    #left half
    y02=y[a02]
    y2=y[:a+1] 
    x2=x[:a+1]
    c2=0
    yp2=[]
    c2 = scipy.mean(y2)
    yp2 = numpy.array(y2 - c2) # neuer array mit um mittelwert veringerten werten: es gibt also jetzt negative und positive werte
    c2 = scipy.mean(c2+abs(yp2))
    yp2 = y2 - c2
    Y2 = numpy.sum(yp2) * (x2[-1] - x2[0]) / len(x2)
    #y02 = yp2.min()
    g2 = Y2 / (numpy.pi * y02)
    a2 = y02 * numpy.pi * g2

        
   #plotten und fitten nur der rechten haelfte
    y03=y[a03]
    y3=y[a-1:] 
    x3=x[a-1:]
    c3=0
    yp3=[]
    c3 = scipy.mean(y3)
    yp3 = numpy.array(y3 - c3) # neuer array mit um mittelwert veringerten werten: es gibt also jetzt negative und positive werte
    c3 = scipy.mean(c3+abs(yp3))
    yp3 = y3 - c3
    Y3 = numpy.sum(yp3) * (x3[-1] - x3[0]) / len(x3)
    #y03 = yp3.min()
    g3 = Y3 / (numpy.pi * y03)
    a3 = y03 * numpy.pi * g3
    #print('x03=',x03,'g3=', g3, a3, c3)
    #print('fitparameters are x0, g, a, c respectively')
    return x02,g2,a2,c2,x03,g3,a3,c3

def DoubleLorentzian_neg( x2,  g2,  a2, c2, x3, g3,a3, c3):
    """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
    c=(c2+c3)/2
    return lambda x: -abs(a2) / numpy.pi * (  abs(g2) / ( (x-x2)**2 + g2**2 )  ) -abs(a3) / numpy.pi * (  abs(g3) / ( (x-x3)**2 + g3**2 )  ) + c


def TripleLorentzian(x1, x2, x3, g1, g2, g3, a1, a2, a3, c):
    """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
    return lambda x: a1 / numpy.pi * (  g1**2 / ( (x-x1)**2 + g1**2 )  ) + a2 / numpy.pi * (  g2**2 / ( (x-x2)**2 + g2**2 )  ) + a3 / numpy.pi * (  g3**2 / ( (x-x3)**2 + g3**2 )  ) + c
setattr(Lorentzian, 'Formula', r'$f(x0,g,a,c;x)=a/\pi (g/((x-x_0)^2+g^2)) + c$')

def trip_lorentz_n14(x1, g, a1, a2, a3, c):
    """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
    n14_split = 2.15*1e6
    x2 = x1 + n14_split
    x3 = x1 + 2*n14_split
    return lambda x: -abs(a1) / numpy.pi * (  g**2 / ( (x-x1)**2 + g**2 )  ) - abs(a2) / numpy.pi * (  g**2 / ( (x-x2)**2 + g**2 )  ) - abs(a3) / numpy.pi * (  g**2 / ( (x-x3)**2 + g**2 )  ) + c

def trip_lorentz_n14_estimator(x, y):
    n14_split = 2.15*1e6    #x in Hz
    dx = abs(x[1]-x[0])
    split_index1 = int(round(n14_split/dx, 0))
    split_index2 = int(round(n14_split*2./dx, 0))
    trip_mean = []
    for i in range(len(y)-split_index2):
        trip_mean.append( (y[i]+y[i+split_index1]+y[i+split_index2])/3. )
    trip_mean = numpy.array(trip_mean)
    c = trip_mean.max()
    x1 = x[trip_mean.argmin()]
    g = 0.5*1e6     #HWHM
    a1 = a2 = a3 = (trip_mean.min()-c) * numpy.pi
    return x1, g, a1, a2, a3, c     

def trip_lorentz_n15(x1, g, a1, a2, c):
    """Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
    n15_split = 3.03*1e6
    x2 = x1 + n15_split
    return lambda x: -abs(a1) / numpy.pi * (  g**2 / ( (x-x1)**2 + g**2 )  ) - abs(a2) / numpy.pi * (  g**2 / ( (x-x2)**2 + g**2 )  ) + c

def trip_lorentz_n15_estimator(x, y):
    n15_split = 3.03*1e6    #x in Hz
    dx = abs(x[1]-x[0])
    split_index = int(round(n15_split/dx, 0))
    trip_mean = []
    for i in range(len(y)-split_index):
        trip_mean.append( (y[i]+y[i+split_index])/2. )
    trip_mean = numpy.array(trip_mean)
    c = trip_mean.max()
    x1 = x[trip_mean.argmin()]
    g = 0.6*1e6     #HWHM
    a1 = a2 = (trip_mean.min()-c) * numpy.pi
    return x1, g, a1, a2, c   
    



def EllipseEstimator(x,y):
    x0 = scipy.mean(x)#average of all x values
    y0 = scipy.mean(y) #average of all x values
    a1 = x.max()- float(x0)
    a2 = float(x0) - x.min()
    a = (a1+a2)/2
    b1 = y.max() - float(y0)
    b2 = float(y0) - y.min()
    b = (b1+b2)/2    
    return x0, y0, a, b 
    
def Ellipse(x0, y0, a, b):
    """Ellipse centered at x0 and y0, semi-major axis a and semi-minor axis b."""
    return lambda x: b * math.sqrt( 1 - ((x-x0)**2)/a**2 ) + y0




def SumOverFunctions( functions ):
    """Creates a factory that returns a function representing the sum over 'functions'.
    'functions' is a list of functions. 
    The resulting factory takes as arguments the parameters to all functions,
    flattened and in the same order as in 'functions'."""
    def function_factory(*args):
        def f(x):
            y = numpy.zeros(x.shape)
            i = 0
            for func in functions:
                n = func.func_code.co_argcount
                y += func(*args[i,i+n])(x)
                i += n
        return f
    return function_factory


def Fit(x, y, Model, Estimator):
    """Perform least-squares fit of two dimensional data (x,y) to model 'Model' using Levenberg-Marquardt algorithm.\n
    'Model' is a callable that takes as an argument the model parameters and returns a function representing the model.\n
    'Estimator' can either be an N-tuple containing a starting guess of the fit parameters, or a callable that returns a respective N-tuple for given x and y."""
    if callable(Estimator):
        return scipy.optimize.leastsq(lambda pp: Model(*pp)(x) - y, Estimator(x,y))[0]
    else:
        return scipy.optimize.leastsq(lambda pp: Model(*pp)(x) - y, Estimator)[0]
 
    
    
class Gaussfit(object):

#    def __init__(self, data):       
#        self.data=data

    def gauss(self, A0, A, x0, wx):
        wx = numpy.float(wx)
        return lambda x: A0+A*numpy.exp(-(((x0-x)/wx)**2)/2)
        
    def moments(self, data):
        #total = data.sum()
        X = numpy.arange(data.size)
        x = (numpy.argmax(data)+len(data)/2.)/2.#sum(X*data)/sum(data)
        wx = numpy.sqrt(abs(sum((X-x)**2*data)/sum(data)))                 
        #wx = numpy.sqrt(abs((numpy.arange(col.size)-1)**2*col).sum()/col.sum())
        #row = data[int(x), :]
        A0 = data.min()   
        A = data.max()-A0
        return A0 , A , x , wx
        
    def fitgaussian(self, data):
        params = self.moments(data)
        errorfunction = lambda p: numpy.ravel(self.gauss(*p)(*numpy.indices(data.shape))-data)
        p, success = scipy.optimize.leastsq(errorfunction, params)
        return p
        
    def Execute(self, data):
        #if data is None:
        #data = self.xydata 
        params = self.fitgaussian(data)       
        A0, A , x , wx = params
        #a  = A**2        
        #return x , y , a
        #return A0**2, A**2, y, x, wx, wy, theta
        return x   
    
class Gaussfit2D(object):

#    def __init__(self, data):       
#        self.data=data

    def gauss(self, A0, A, x0, y0, wx, wy):
        wx = numpy.float(wx)
        wy = numpy.float(wy)
        #def f(x,y):
        #    x = (x-x0)*numpy.cos(theta) + (y-y0)*numpy.sin(theta)
        #    y = (x-x0)*numpy.sin(theta) + (y-y0)*numpy.cos(theta)
        #    return A0**2+A*A*numpy.exp(-((x/wx)**2+(y/wy)**2)/2)
        #return f
        return lambda x,y: A0**2+A*A*numpy.exp(-(((x0-x)/wx)**2+((y0-y)/wy)**2)/2)
        
    def moments(self, data):
        total = data.sum()
        X, Y = numpy.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        wx = numpy.sqrt(abs((numpy.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        wy = numpy.sqrt(abs((numpy.arange(row.size)-x)**2*row).sum()/row.sum())
        A = numpy.sqrt(data.max())
        A0 = numpy.sqrt(data.min())
        return A0 , A , x , y , wx , wy
        
    def fitgaussian(self, data):
         
        params = self.moments(data)
        errorfunction = lambda p: numpy.ravel(self.gauss(*p)(*numpy.indices(data.shape))-data)
        p, success = scipy.optimize.leastsq(errorfunction, params)
        return p
        
    def Execute(self, data):
        #if data is None:
        #data = self.xydata 
        params = self.fitgaussian(data)       
        A0, A , y , x , wx , wy = params
        #a  = A**2        
        #return x , y , a
        #return A0**2, A**2, y, x, wx, wy, theta
        return x , y


