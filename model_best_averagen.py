from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from pylab import *
import scipy.stats as ss
import lmfit as lm
from scipy.ndimage import uniform_filter1d

# Load data
data1= np.loadtxt("/Users/luna/Documents/GitHub/409 muon/Oct14.txt")[5:-3] 
print("here is the sum", np.sum(data1))



def expon(x, a, b, d):
    return a*e**((-(1/b)*x))+ d

# Bining number
tau = []
tau_uncertainty = []
x2 = np.arange(5,2045,1)
ppot_arr = []
sum_data_total = []
time2 = (x2+173.7726257888435)/259.6537265639224


N = 31+5

num_bins = len(data1)//N
x= np.arange(5,2045,N)

time1 = (x+173.7726257888435)/259.6537265639224

#The data will also be less. need to take this into account
sum_data = []

for i, bin in enumerate(range(num_bins)):
    if i==0:
        data = data1[i:N]
        sum_data1 = np.sum(data)
        sum_data.append(sum_data1)
    else:
        data = data1[N*i:N*i+N]
        sum_data1 = np.sum(data)
        sum_data.append(sum_data1)
sum_data_total.append(sum_data)
emodel=lm.Model(expon)
params = emodel.make_params(a=78, b=4, d=0.44)
ymodel = emodel.eval(params,  x= time1[0:len(sum_data)])

yerr1= np.sqrt(sum_data)

popt, pcov = scipy.optimize.curve_fit(expon, time1[0:len(sum_data)], sum_data, sigma = yerr1)
tau.append(popt[1])
tau_uncertainty.append(np.sqrt(np.diag(pcov))[1])
ppot_arr.append(popt)

weights = []
for i,element in enumerate(yerr1):
    if element ==0:
        weights.append(1)
    else:
        weights.append(1/element)
emodel=lm.Model(expon)
params = emodel.make_params(a=78, b=4, d=0.44)
ymodel = emodel.eval(params,  x= time1[0:len(sum_data)])


result=emodel.fit(sum_data, params, x= time1[0:len(sum_data)],weights=weights[0:len(sum_data)])
print(result.fit_report())
print(popt)
# para =  ppot_arr[index[0][0]]
y_fit_muon = expon(time1[0:len(sum_data)], result.params['a'].value, result.params['b'].value, result.params['d'].value)
lifetime = result.params['b'].value
lifetime_un = result.params['b'].stderr
# print(np.shape(time1[0:len(sum_data)]))
# print(np.shape(y_fit_muon))
print(result.params['b'].value)
print(result.params['b'].stderr)


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax1 = plt.subplots()

ax1.bar(time1[0:len(sum_data)],  sum_data_total[0], edgecolor=None, width = 0.11)
ax1.errorbar(time1[0:len(sum_data)], sum_data, yerr= np.sqrt(sum_data),fmt = "r.",  elinewidth=1, markersize=1, capsize=3, color="k")
ax1.plot( time1[0:len(sum_data)], y_fit_muon,'r-', label = r"$\tau_\mu$ = {:.2f} $\pm$ {:.2f} $\mu s$".format(lifetime, lifetime_un))
ax1.set_xlabel(r"Time $\mu s$",fontsize = 18)
ax1.set_ylabel(r"Counts",fontsize = 18)
# ax1.rcParams.update({'font.size': 22})
l, b, h, w = .5, .5, .4, .4
ax2 = fig.add_axes([l, b, w, h])
ax2.bar(time2, data1, label = "raw data", edgecolor=None, width = 0.02)
ax2.set_xlabel(r"Time $\mu s$",fontsize = 18)
ax2.set_ylabel(r"Counts",fontsize = 18)
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(16)
for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
	label.set_fontsize(16)
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper left')
plt.show()
# plt.bar(time1[0:len(sum_data)],  sum_data_total[0], edgecolor=None, width = 0.11)
# plt.errorbar(time1[0:len(sum_data)], sum_data, yerr= np.sqrt(sum_data),fmt = "r.",  elinewidth=1, markersize=1, capsize=3, color="k")
# plt.plot( time1[0:len(sum_data)], y_fit_muon,'r-', label = r"$\tau_\mu$ = {:.2f} $\pm$ {:.2f} $\mu s$".format(lifetime, lifetime_un))
# plt.bar(time2, data1, label = "raw data", edgecolor=None, width = 0.02)
# plt.ylabel(r"Counts")
# plt.xlabel(r"Time $\mu s$")
# plt.show()