import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from astropy.constants import c
from astropy import units as u

# Local file path
file_path = 'Pantheon+SH0ES.dat'

# Load the file

data = pd.read_csv('Pantheon+SH0ES.dat', delim_whitespace = True, comment = '#')


# See structure

# Show the first 5 rows of the dataset

#data.head()
print(data.columns)


# Filter for entries with usable data based on the required columns

# Drop rows with missing values in required columns

data_clean = data.dropna(subset=['zHD', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG'])

# Extract relevant columns as NumPy arrays

z = data_clean['zHD'].values
mu = data_clean['MU_SH0ES'].values
mu_err = data_clean['MU_SH0ES_ERR_DIAG'].values

# Confirm number of valid data points
print(f"Number of usable data points: {len(z)}")

#Plot the Hubble Diagram

plt.figure(figsize=(8, 6))

# Plot observed data with error bars
plt.errorbar(z, mu, yerr=mu_err, fmt='o', markersize=3,
 capsize=2, alpha=0.6, label='Supernovae Ia (Pantheon+SH0ES)')

# Use logarithmic scale for redshift
plt.xscale('log')

# Labels and title
plt.xlabel("Redshift (z)")
plt.ylabel("Distance Modulus (mu)")
plt.title("Hubble Diagram")

# Formatting
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Define the E(z) for flat LCDM
def E(z, Omega_m):
   return np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))


# Luminosity distance in Mpc, try using scipy quad to integrate.
def luminosity_distance(z, H0, Omega_m):
    integrand = lambda z_prime: 1.0 / E(z_prime, Omega_m)

    # Numerically integrate from 0 to z
    integral, _ = quad(integrand, 0, z)

    # Speed of light in km/s
    c_kms = c.to('km/s').value

    # Compute luminosity distance
    d_L = (1 + z) * (c_kms / H0) * integral  # in Mpc
    return d_L


# Theoretical distance modulus, use above function inside mu_theory
 #to compute luminosity distance
def mu_theory(z, H0, Omega_m):
    d_L = luminosity_distance(z, H0, Omega_m)
    return 5 * np.log10(d_L) + 25

# Initial guess: H0 = 70, Omega_m = 0.3
#p0 = [H guess, omega guess]

# Write a code for fitting and taking error out of the parameters


def mu_theory_vectorized(z, H0, Omega_m):
    return np.array([mu_theory(zi, H0, Omega_m) for zi in z])

# Initial guess: H0 = 70 km/s/Mpc, Omega_m = 0.3
p0 = [70, 0.3]

# Perform curve fitting
popt, pcov = curve_fit(mu_theory_vectorized, z, mu, sigma=mu_err,
 p0=p0, absolute_sigma=True)

# Extract best-fit parameters and their uncertainties
H0_fit, Omega_m_fit = popt
H0_err, Omega_m_err = np.sqrt(np.diag(pcov))

# Print the results
print(f"Fitted H0 = {H0_fit:.2f} ± {H0_err:.2f} km/s/Mpc")
print(f"Fitted Omega_m = {Omega_m_fit:.3f} ± {Omega_m_err:.3f}")

# Write the function for age of the universe as above

def age_of_universe(H0, Omega_m):

   def integrand(z):
        return 1.0 / ((1 + z) * E(z, Omega_m))


   integral, _ = quad(integrand, 0, 1000)


   H0_SI = H0 / (3.086e19)


   age_sec = integral / H0_SI


   age_gyr = age_sec / (60 * 60 * 24 * 365.25 * 1e9)

   return age_gyr

t0 = age_of_universe(H0_fit, Omega_m_fit)
print(f"Estimated age of Universe: {t0:.2f} Gyr")


# Write the code to find residual by computing mu_theory and then plot
#mu_model = mu_theory(your theory)

# Vectorized version of mu_theory using best-fit parameters
mu_model = np.array([mu_theory(zi, H0_fit, Omega_m_fit) for zi in z])

# Compute residuals: observed - model
residuals = mu - mu_model

# Plot residuals vs redshift
plt.figure(figsize=(8, 5))
plt.axhline(0, color='black', linestyle='--', alpha=0.7)
plt.scatter(z, residuals, s=10, alpha=0.6, color='darkblue')
plt.xscale('log')

plt.xlabel("Redshift (z)")
plt.ylabel("Residual (mu_obs - mu_model)")
plt.title("Residuals of Supernova Distance Modulus Fit")
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# Plot Hubble diagram with model fit

plt.figure(figsize=(8, 6))

# Plot observed data with error bars
plt.errorbar(z, mu, yerr=mu_err, fmt='o', markersize=3, capsize=2,
 alpha=0.5, label='Supernovae Ia (Observed)')

# Plot model fit using best-fit H0 and Omega_m
z_sorted = np.sort(z)
mu_model_sorted = [mu_theory(zi, H0_fit, Omega_m_fit) for zi in z_sorted]
plt.plot(z_sorted, mu_model_sorted, color='red', linewidth=2,
 label='ΛCDM Model Fit')

# Log scale
plt.xscale('log')
plt.xlabel("Redshift (z)")
plt.ylabel("Distance Modulus (mu)")
plt.title("Hubble Diagram with Model Fit")
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# Try fitting with this fixed value

def mu_fixed_Om(z, H0):
    return np.array([mu_theory(zi, H0, Omega_m=0.3) for zi in z])

# Initial guess for H0
p0_fixed = [70]


popt_fixed, pcov_fixed = curve_fit(mu_fixed_Om, z, mu, sigma=mu_err,
 p0=p0_fixed, absolute_sigma=True)

# Extract fitted H0 and its error
H0_fixed = popt_fixed[0]
H0_fixed_err = np.sqrt(pcov_fixed[0][0])

# Print the result
print(f"Fitted H0 with Omega_m = 0.3: {H0_fixed:.2f} ± {H0_fixed_err:.2f} km/s/Mpc")


# Split the data for the three columns and do the fitting again and see

# Define the redshift split point
z_split = 0.1

# Split data into low-z and high-z samples
low_z_mask = z < z_split
high_z_mask = z >= z_split

z_low, mu_low, mu_err_low = z[low_z_mask], mu[low_z_mask], mu_err[low_z_mask]
z_high, mu_high, mu_err_high = z[high_z_mask], mu[high_z_mask], mu_err[high_z_mask]

# Fitting function (Omega_m fixed at 0.3)
def mu_fixed_Om(z, H0):
    return np.array([mu_theory(zi, H0, Omega_m=0.3) for zi in z])

# Fit low-z sample
H0_low, pcov_low = curve_fit(mu_fixed_Om, z_low, mu_low, sigma=mu_err_low,
p0=[70], absolute_sigma=True)

# Fit high-z sample
H0_high, pcov_high = curve_fit(mu_fixed_Om, z_high, mu_high, sigma=mu_err_high,
p0=[70], absolute_sigma=True)

# Print results
print(f"Low-z (z < {z_split}): H₀ = {H0_low[0]:.2f} km/s/Mpc")
print(f"High-z (z ≥ {z_split}): H₀ = {H0_high[0]:.2f} km/s/Mpc")
