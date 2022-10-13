from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from pylab import *
import uncertainties as uc
import scipy.odr as od


data = np.loadtxt("/Users/luna/Documents/GitHub/409 muon/Calib sept 28/sept_28_calib_10.txt")
index1= np.nonzero(data) #get all non_zero_index
import lmfit as lm
# print(index)

# for index in index1[0]:
#     print(f"here is index {index} with data {data[index]}")

data1 = np.array([85, 342, 600, 858, 1116, 1373, 1631, 1889])
data2 = np.array([85, 342, 600, 858, 1116, 1374, 1632, 1889])
data3 = np.array([85, 342, 600, 858, 1116, 1374, 1632, 1890])
data4 = np.array([85, 342, 600, 858, 1116, 1373, 1631, 1889])
data5 = np.array([84, 340, 596, 852, 1108, 1363, 1619, 1875])
data6 = np.array([87, 349, 611, 873, 1135, 1397, 1659, 1921])
data7 = np.array([87, 349, 611, 873, 1135, 1397, 1659, 1921])
data8 = np.array([87, 349, 611, 873, 1135, 1397, 1659, 1921])
data9 = np.array([87, 349, 611, 873, 1135, 1397, 1659, 1921])
data10 = np.array([87, 349, 611, 873, 1135, 1397, 1659, 1921])

total = np.array([data1,data2,data3, data4,data5,data6,data7,data8,data9,data10])

def linear(x, m, b):
    return m*x+b

lmodel = lm.Model(linear)


standard_dev = []
standard_error = []
mean1 = []
for i in range(8):
    sdtev = np.std(total[:,i])
    avg = np.mean(total[:,i])
    mean1.append(avg)
    standard_dev.append(sdtev)
    standard_error.append(sdtev*np.sqrt(100))


time1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])


params = lmodel.make_params(m=200, b=-170)

ymodel = lmodel.eval(params, x=time1)
# error_fake = np.ones(8)
yerr = standard_error
# yerr = error_fake
weights = [1./i for i in yerr]


result= lmodel.fit(mean1, params, x=time1, weights=weights)
print(result.fit_report())


def f1(x, m, b):
    return m*(x)+b

popt, pcov = scipy.optimize.curve_fit(f1,time1, mean1, sigma=standard_dev) 
y_fit2 = f1(time1,*popt)
print(*popt)
print (np.sqrt(np.diag(pcov)))
# plt.title("Pluse height to time conversion")
plt.xlabel(r"Time ($\mu s$)")
plt.ylabel(r"Pulse height (keV)")
# plt.plot(time1, mean1, 'b.', markersize=1)
# plt.errorbar(time1,mean1, yerr=standard_error,fmt = ".",  markersize=0, capsize=2, color="k")

plt.plot(time1, result.best_fit, 'r', label = "y = ({:.2f} $\pm$ {:.2f}) x+ ({:.2f} $\pm$ {:.2f})".format(popt[0],np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1]))
plt.plot(time1, mean1, 'b.', markersize=2)
grid(color='0.95', linestyle='-', linewidth=1)
plt.errorbar(time1,mean1, yerr=standard_dev,fmt = ".",  markersize=0, capsize=2, color="k")
# plt.legend()
plt.show()


