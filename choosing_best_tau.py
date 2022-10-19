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
N_arr = np.arange(5,50,1)
for n in range(5,50):
    N = n

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
    # tau.append(result[])
    # tau_uncertainty.append(np.sqrt(np.diag(pcov))[1])
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
    tau.append(result.params['b'].value)
    tau_uncertainty.append(result.params['b'].stderr)
    print(result.fit_report())

index = np.where(tau_uncertainty==np.nanmin(tau_uncertainty))
print(index)
para =  ppot_arr[index[0][0]]
y_fit_muon = expon(time1[0:len(sum_data)], para[0], para[1], para[2])
lifetime = tau[index[0][0]]
lifetime_un = tau_uncertainty[index[0][0]]
print(lifetime,lifetime_un)

# special = np.zeros(len(tau))
N_2 = 36
# plt.bar(time1[0:len(sum_data)],  sum_data_total[index[0][0]], edgecolor=None, width = 0.12)
# plt.errorbar(time1[0:len(sum_data)], sum_data, yerr= np.sqrt(sum_data),fmt = "r.",  elinewidth=1, markersize=1, capsize=3, color="k")
plt.plot(N_arr, tau_uncertainty,'b.')
plt.plot(N_2, tau_uncertainty[31], 'r.')
print(tau)
# # plt.plot(time2, data1,'g', label = "raw data")
plt.xlabel(r"Bin size")
plt.ylabel(r"$\tau_\mu$ uncertainty")
plt.legend()
plt.show()

