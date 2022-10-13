"""
The program SimpleFit.py uses curve_fit to do a linear fit to data
with NO uncertainty estimate for the data.
"""
# The following two commands import the program modules needed to
# run this program.
from pylab import *
from scipy.optimize import curve_fit

# This function defines the function to be fit. In this case a linear
# function with slope 'b' and intercept 'a'.
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

# This line reads the data from a the file `FakeData_with_error.txt` and 
# places it in three arrays.
xdata,ydata,d_y = loadtxt('FakeData_with_error.txt',unpack=True)


# This line calls the curve_fit function. It returns two arrays.
# 'a_fit' contains the best fit parameters and 'cov' contains
# the covariance matrix.
a_fit,cov=curve_fit(linearFunc,xdata,ydata,sigma=d_y,absolute_sigma=True)

# The next four lines define variables for the slope, intercept, and
# there associated uncertainties d_slope and d_inter. The uncertainties
# are computed from elements of the covariance matrix.
inter = a_fit[0]
slope = a_fit[1]
d_inter = sqrt(cov[0][0])
d_slope = sqrt(cov[1][1])

# Create a graph showing the data.
errorbar(xdata,ydata,yerr=d_y,fmt='r.',label='Data')

# Compute a best fit line from the fit intercept and slope.
yfit = inter + slope*xdata

# Create a graph of the fit to the data. We just use the ordinary plot
# command for this.
plot(xdata,yfit,label='Fit')

# Display a legend, label the x and y axes and title the graph.
legend()
xlabel('x')
ylabel('y')

# Save the figure to a file
savefig('FakeDataPlot_with_error.png')

# Show the graph in a new window on the users screen.
show()

# Print the best fit values for the slope and intercept. These print
# statments illustrate how to print a mix of strings and variables.
print(f'The slope = {slope}, with uncertainty {d_slope}')
print(f'The intercept = {inter}, with uncertainty {d_inter}')

# We can estimate the goodness of fit for a fit to data with uncertainties by
# computing the reduced chi-squared statistic. For a good fit it should be
# approximatly equal to one.
chisqr = sum((ydata-linearFunc(xdata,inter,slope))**2/d_y**2)
dof = len(ydata) - 2
chisqr_red = chisqr/dof
print(f'Reduced chi^2 = {chisqr_red}')