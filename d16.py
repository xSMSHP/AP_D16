import re
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import numpy as np


class A1:
    def __init__(self, file, name):
        self.name = name
        self.x = []
        self.U = []
        
        for line in file:
            l = re.split(',', line)
            self.x.append(float(l[1]))
            self.U.append(np.log(float(l[0])))
    
    def plot(self):
        coefU = np.polyfit(self.x, self.U, 1)
        poly1d_fnU = np.poly1d(coefU) 
        print(poly1d_fnU)
        print('Konstante K = ' + str(-coefU[0]))
        
        plt.figure()
        plt.plot(self.x, self.U, 'ro')
        plt.plot(np.linspace(0, 5, 100), poly1d_fnU(np.linspace(0, 5, 100)), 'r-')
        plt.legend(['Messwerte', 'Ausgleichsgerade'])
        plt.xlabel('x [cm]')
        plt.ylabel('ln(U)')
        plt.grid(True)
        plt.autoscale(True)
        plt.savefig(self.name + '.png', dpi = 900)
        plt.show()

class Extremwert:
    def __init__(self):
        self.HP = 0
        self.TP = 0
        self.HPU = 0
        self.TPU = 0
    
    def dist(self):
        return abs(self.HP - self.TP)

class A2:
    def __init__(self, file, name):
        self.name = name
        self.x0 = []
        self.extr = []
        self.x = []
        self.U = []
        
        for i, line in enumerate(file):
            l = re.split(',', line)
            if i % 2 == 0:
                if i is not 0:
                    self.x0.append(self.extr[-1].dist())
                self.extr.append(Extremwert())
            if l[0] == 'TP':
                self.extr[-1].TP = float(l[2])
                self.extr[-1].TPU = float(l[1])
            else:
                self.extr[-1].HP = float(l[2])
                self.extr[-1].HPU = float(l[1])
            self.x.append(float(l[2]))
            self.U.append(float(l[1]))
        
        self.x0.append(self.extr[-1].dist())
    
    def average(self):
        out = 0
        for x in self.x0:
            out += x
        return out / len(self.x0)
    
    def variance(self):
        out = 0
        av = self.average()
        for x in self.x0:
            out += (av - x) ** 2
        return np.sqrt(out / (len(self.x0) - 1))
    
    def plot(self):
        HP = []
        TP = []
        for ex in self.extr:
            HP.append(ex.HPU)
            TP.append(ex.TPU)
        HPav = 0
        for h in HP:
            HPav += h
        HPav /= len(HP)
        TPav = 0
        for t in TP:
            TPav += t
        TPav /= len(TP)
        
        guess_mean = np.mean(self.U)
        guess_phase = 0
        amp = (HPav - TPav) / 2
        
        optimize_func = lambda x: amp * np.sin(np.pi / self.average() * t + x[0]) + guess_mean - self.U
        est_phase = leastsq(optimize_func, [guess_phase])[0]
        
        print('Amplitude: ' + str(amp) + '  Phase: ' + str(est_phase) + '   Höhe: ' + str(guess_mean))
        
        plt.figure()
        plt.plot(self.x, self.U, 'ro')
        plt.plot(np.linspace(30, 51, 1000), amp * np.sin(np.pi / self.average() * np.linspace(30, 51, 1000) + est_phase) + guess_mean, 'b-')
        plt.vlines(30, 0, 12, 'k')
        plt.xlabel('x [cm]')
        plt.ylabel('U [V]')
        plt.legend(['Messwerte', 'Extrapolation','Metallplatte'], loc = 5)
        plt.grid(True)
        plt.autoscale(True)
        plt.savefig(self.name + '.png', dpi = 900)
        plt.show()

class A3:
    def __init__(self, file, d):
        self.dx = []
        self.d = d
        
        extr = 0
        for i, line in enumerate(file):
            l = re.split(',', line)
            if i % 2 == 0:
                if i is not 0:
                    self.dx.append(extr.dist())
                extr = Extremwert()
            if l[0] == 'TP':
                extr.TP = float(l[2]) - float(l[3])
            else:
                extr.HP = float(l[2]) - float(l[3])
        self.dx.append(extr.dist())
        
        out = 0
        for x in self.dx:
            out += x
        out /= len(self.dx)
        
        var = 0
        for x in self.dx:
            var += (out - x) ** 2
        var = np.sqrt(var / (len(self.dx) - 1))
            
        print('dx = ' + str(out) + ' n = ' + str(1 + out / self.d) + '  delta dx = ' + str(var))

class A4:
    def __init__(self, file, name):
        self.name = name
        self.U = []
        self.g = []
        
        for line in file:
            l = re.split(',', line)
            self.U.append(float(l[0]))
            self.g.append(90 - float(l[1]))
            
    def plot(self):
        coefU = np.polyfit(self.g, self.U, 2)
        poly1d_fnU = np.poly1d(coefU) 
        maxi = poly1d_fnU.deriv().r
        g = maxi[maxi.imag==0].real[0]
        b = 39.8 + 7 + 90 - g
        f = 1 / (1 / b + 1 / g)
        print(poly1d_fnU)
        
        print('g = ' + str(g) + ' b = ' + str(b) + ' f = ' + str(f) + ' n = ' + str(1 + 15 / f))
        
        plt.figure()
        plt.plot(self.g, self.U, 'bo')
        plt.plot(np.linspace(20, 85, 1000), poly1d_fnU(np.linspace(20, 85, 1000)), 'b-')
        plt.plot([g], poly1d_fnU(g), 'ro')
        plt.xlabel('g [cm]')
        plt.ylabel('U [V]')
        plt.legend(['Messwerte', 'Ausgleichskurve', 'Maximum'])
        plt.grid(True)
        plt.autoscale(True)
        plt.savefig(self.name + '.png', dpi = 900)
        plt.show()

class A5:
    def __init__(self, file, name, n, lam, d):
        self.name = name
        self.d = d / 2
        self.n = n
        self.lam = lam
        self.alpha = []
        self.U = []
        
        for line in file:
            l = re.split(',', line)
            self.alpha.append(float(l[1]))
            self.U.append(float(l[0]))
    
    def plot(self):
        plt.figure()
        plt.plot(self.alpha, self.U, 'ro', label = 'Messwerte')
        plt.plot(self.alpha, self.U, 'r-')
        plt.vlines(np.arcsin(np.arange(0, self.n, 1) * self.lam / self.d) * 360 / (2 * np.pi), 0, 6.5, color = 'blue', label = 'theoretische Maxima')
        plt.vlines(np.arcsin((2 * np.arange(1, self.n, 1) - 1) * self.lam / (2 * self.d)) * 360 / (2 * np.pi), 0, 6.5, color = 'green', label = 'theoretische Minima')
        plt.xlabel(r'Winkel $\alpha$ [$^\circ$]')
        plt.ylabel('U [V]')
        plt.legend()
        plt.grid(True)
        plt.autoscale(True)
        plt.savefig(self.name + '.png', dpi = 900)
        plt.show()

class A6:
    def __init__(self, file, name):
        self.name = name
        self.theta = []
        self.U_base = []
        self.U = []
        
        switch = False
        for line in file:
            l = re.split(',', line)
            if len(l) == 1:
                switch = True
                continue
            if switch:
                self.U.append(float(l[0]))
            else:
                self.theta.append(float(l[1]))
                self.U_base.append(float(l[0]))
    
    def plot(self):
        print('Maxima: ' + str(self.theta[4]) + ' , ' + str(self.theta[12]))
        plt.figure()
        plt.plot(self.theta, self.U_base, 'ro', label = 'Direktstrahl')
        plt.plot(self.theta, self.U, 'bo', label = 'mit Gitter')
        plt.axvspan(self.theta[2], self.theta[5], color = 'purple', alpha = 0.5)
        plt.axvspan(self.theta[10], self.theta[13], color = 'purple', alpha = 0.5)
        plt.xlabel(r'Winkel $2\theta$ [$^\circ$]')
        plt.ylabel('U [V]')
        plt.legend()
        plt.grid(True)
        plt.autoscale(True)
        plt.savefig(self.name + '.png', dpi = 900)
        plt.show()
            
a1 = A1(open('a1.txt', 'r'), 'a1')
a1.plot()

a2 = A2(open('a2.txt', 'r'), 'a2')
print('lambda = ' + str(2 * a2.average()) + ' ; delta lambda = ' + str(a2.variance()))
a2.plot()

a3 = A3(open('a3.txt', 'r'), 2)

a4 = A4(open('a4.txt', 'r'), 'a4')
a4.plot()

a5 = A5(open('a5.txt', 'r'), 'a5', 3, 2 * a2.average(), 8.5)
a5.plot()

a6 = A6(open('a6.txt', 'r'), 'a6')
a6.plot()
