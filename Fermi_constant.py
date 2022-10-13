import numpy as np
import lmfit as lm
import uncertainties as uc
import matplotlib.pyplot as plt

h = 6.582* 10**(-25) #in GeV.s
muon_mass = 1.057e-1 #GeV
c = 3*10**8 #in s
tau = 2.197* 10**-6 # in s


tau = uc.ufloat(2.197e-6, 0.07e-6)
muon_mass = uc.ufloat(1.057e-1, 2.3e-8)
# G_f = tau/muon_mass
G_f = (h*192*(np.pi)**3 /(tau*(muon_mass)**5))**(1/2)
print(G_f)