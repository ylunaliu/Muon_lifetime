import numpy as np
import lmfit as lm
import uncertainties as uc
import matplotlib.pyplot as plt

y = uc.ufloat(565.102539, 18.6524844)
m = uc.ufloat(259.653727,  0.03254161)
# b = uc.ufloat( -173.772626, 0.05266177)

x = (y)/m
# data1= np.loadtxt("/Users/luna/Documents/GitHub/409 muon/run1_26.txt")[6:] 


print(x)