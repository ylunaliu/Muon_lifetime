from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from pylab import *
import scipy.stats as ss
import lmfit as lm
from scipy.ndimage import uniform_filter1d

data1= np.loadtxt("/Users/luna/Documents/GitHub/409 muon/Oct_12_data.txt")[5:-3] 
data2= np.loadtxt("/Users/luna/Documents/GitHub/409 muon/Oct7-data.txt")[5:-3] 
print("here is the sum", np.sum(data1))
print("here is the sum", np.sum(data2))
N = 36
num_bins = len(data1)//N
x= np.arange(5,2045,N)
time1 = (x+173.7726257888435)/259.6537265639224
print(len(x))
#The data will also be less. need to take this into account
averaged_data_total = []
yerror = []

for i, bin in enumerate(range(num_bins)):
    if i==0:
        data = data1[i:N]
        averaged_data = np.average(data)
        averaged_data_total.append(averaged_data)
        stdev1 = np.std(data)
        yerror.append(stdev1)
    else:
        data = data1[N*i:N*i+N]
        averaged_data = np.average(data)
        averaged_data_total.append(averaged_data)
        stdev1 = np.std(data)
        yerror.append(stdev1)


def expon(x, a, b, d):
    return a*e**((-(1/b)*x))+ d

emodel=lm.Model(expon)
params = emodel.make_params(a=78, b=4, d=0.44)
ymodel = emodel.eval(params,  x= time1)


yerr1= map(lambda x: x, yerror)

weights = []
for i,element in enumerate(yerr1):
    if element ==0:
        weights.append(1)
    else:
        weights.append(1/element)


result=emodel.fit(averaged_data_total[:], params, x= time1[0:len(averaged_data_total)],weights=weights[0:len(averaged_data_total)])
print(result.fit_report())

# plt.plot(time2, data1,label = "raw data")
# plt.plot(time2, data2,'g', label = "raw data")
# plt.plot(time2, data1,label = "raw data")
# plt.plot(time1[20:], averaged_data_total[20:], 'k-', label="averaged data with N=5")
# plt.errorbar(time1,averaged_data_total, yerr= yerror,fmt = "r.",  elinewidth=1, markersize=1, capsize=3, color="k")
# plt.plot(averaged_data_total, np.sqrt(averaged_data_total), '.')
plt.plot(averaged_data_total, yerror, '.', label = r"Averaged N = {:.0f}".format(N))
plt.plot(averaged_data_total, np.sqrt(averaged_data_total), label = r"Poisson error $\sigma = \sqrt{N}$")
plt.ylabel(r"Standard deviation $\sigma$")
plt.xlabel(r"Counts")
# # plt.plot(time1[20:], result.best_fit, 'r')

plt.legend()
plt.show()