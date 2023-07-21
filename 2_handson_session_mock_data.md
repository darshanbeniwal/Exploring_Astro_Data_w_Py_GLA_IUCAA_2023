## Install the required packages
```python

!pip install lmfit

```
 
## Load the required packages
```python

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import PolynomialModel
from lmfit.models import ExpressionModel

```

## Load Mock dataset
```python

x=np.array((1.0,2.0,3.0,4.0,5.0))
y=np.array((2.3,4.1,6.2,8.1,10.0))
yerr=np.array((0.08, 0.12, 0.2 , 0.16, 0.28))

```
## Plot datapoint
```python

plt.errorbar(x,y,yerr,fmt='b.')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

```
## Run lmfit package
```python

mod = ExpressionModel('a+b*x')
pars = mod.make_params(a=0, b=0)
result = mod.fit(y, pars, x=x, weights=1.0/yerr)

```

## Print results
```python

# Assuming you have already obtained the `result` object from the fitting process

# Extracting the parameters and their uncertainties
a_fit = result.params['a'].value
b_fit = result.params['b'].value
sig_a = result.params['a'].stderr  # Uncertainty as a percentage of the parameter value
sig_b = result.params['b'].stderr  # Uncertainty as a percentage of the parameter value

# Printing the values
print("Fitted a =", a_fit)
print("Fitted b =", b_fit)
print("Uncertainty in a =", sig_a)
print("Uncertainty in b =", sig_b)


```

## Plot Contours
```python


def model_func(x, a, b):
    return a + b * x
a_error, b_error = sig_a,sig_b

# Calculate the confidence level contours
delta_chisq = 2.30  # 68% confidence level (2 degrees of freedom)
a_vals = np.linspace(a_fit - 5 * a_error, a_fit + 5 * a_error, 100)
b_vals = np.linspace(b_fit - 5 * b_error, b_fit + 5 * b_error, 100)
A, B = np.meshgrid(a_vals, b_vals)
chi_sq = np.zeros_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        model_vals = model_func(x, A[i, j], B[i, j])
        chi_sq[i, j] = np.sum(((y - model_vals) / yerr) ** 2)

# Plot the 68% confidence level contour
plt.contour(A, B, chi_sq, levels=[delta_chisq], colors='blue', linestyles='dashed', label='68% CL')

# Calculate the confidence level contours for 95% (6.18 for 2 degrees of freedom)
delta_chisq_95 = 6.18
plt.contour(A, B, chi_sq, levels=[delta_chisq_95], colors='green', linestyles='dashed', label='95% CL')
plt.xlabel('a')
plt.ylabel('b')
plt.legend()
plt.show()

# Print the fitted parameters and their errors
print("Fitted a =", a_fit, "+/-", a_error)
print("Fitted b =", b_fit, "+/-", b_error)


```

## Plot mock data best fit curve and Save it
```python
xfine = np.arange(0,5.2,0.001)
ypred = result.eval(x=xfine)
dely = result.eval_uncertainty(x=xfine, sigma=1)
plt.errorbar(x, y, yerr, linewidth=2, color='g', ls='none', mfc='r', marker='+', markersize=6)
plt.plot(xfine, ypred, 'k--',label='Best Fitted Line of Linear model: $a+bx$')
plt.fill_between(xfine, ypred-2*dely, ypred+2*dely, color="r",label="$2\sigma$ Confidence Level")
plt.fill_between(xfine, ypred-dely, ypred+dely, color="#ABABAB",label="$1\sigma$ Confidence Level")
plt.title('$\Lambda$CDM Model fitting with $H(z)$ measurements')
plt.xlim(0,5.2)
plt.grid(False)
plt.xlabel('$z$')
plt.ylabel('$H(z)$')
plt.legend(loc="upper left")
plt.savefig('Linear_model_fit_mock_data_u_lmfit.pdf', format='pdf', dpi=1200)

```

## =================================================================
