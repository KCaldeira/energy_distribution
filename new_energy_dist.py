# translate version of Mathematica code

#%% Import packages

import numpy as np
import pandas as pd
from datetime import datetime

from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit

from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

import math
import statsmodels.api as sm
import time
import platform, os
from scipy.special import beta as beta_func, betainc

#%% compute country-level elasticity of energy use


def compute_elasticity_of_energy_use(excel_input_data):
    """
    Computes the elasticity of energy use based on input data.

    This elasticity is computed at country-level by doing a population-weighted least-squares linear regression
    of the log of mean country-level per-capita energy use as a function of the log of mean 
    country-level per-capita GDP, subject to the constraint that the sum of the product of population
    and per-capita gdp raised to the gamma parameter is equal to global energy use, where gamma is
    the slope of that linear regression.

    Parameters:
        excel_input_data: pandas.DataFrame
            Input data where rows correspond to countries and columns to various parameters:
            - Column 5: Population (2019, in thousands)
            - Column 6: GDP (2019, in 2017 dollars)
            - Column 7: Energy use (in TJ/year).

    Returns:
        tuple:
            - Elasticity coefficient (slope of the regression line).
            - Adjusted R-squared value.
            - Standard error of the slope parameter.
    """
    # Extract relevant data columns
    pop = 1000 * excel_input_data.iloc[:, 4].to_numpy()  # Population in number of people (2019)
    gdp = excel_input_data.iloc[:, 5].to_numpy()         # GDP in 2017 dollars (2019)
    energy = (10**9 / 8760) * excel_input_data.iloc[:, 6].to_numpy()  # Energy use in kW (average)

    model_params = fit_population_weighted(pop, energy/pop, gdp/pop)

    elasticity_coefficient = model_params[0]
    r_squared = model_params[2]

    return elasticity_coefficient, r_squared

def fit_population_weighted(pop, per_capita_energy, per_capita_gdp):
    """
    Fits the model:
        log(per_capita_energy) = a * log(per_capita_gdp) + b
    subject to the constraint:
        sum(pop * per_capita_energy) = exp(b) * sum(pop * per_capita_gdp^a)
    
    The parameter b is eliminated using:
        b = log(sum(pop * per_capita_energy)) - log(sum(pop * per_capita_gdp^a))
    
    The function returns the optimal parameters (a, b) along with the weighted R^2,
    where the weighted R^2 is computed on the log-transformed per_capita_energy data.
    
    Parameters:
        pop:              numpy array of population weights.
        per_capita_energy: numpy array of per-capita energy values (must be positive).
        per_capita_gdp:    numpy array of per-capita GDP values (must be positive).
    
    Returns:
        a_best, b_best, R2: optimal parameters and the weighted fraction of variance explained.
    """
    # Define the objective function to minimize over a
    def objective(a):
        # Compute b from the constraint for the given a
        b = np.log(np.sum(pop * per_capita_energy)) - np.log(np.sum(pop * per_capita_gdp**a))
        # Compute the residuals in the log-space
        residuals = np.log(per_capita_energy) - a * np.log(per_capita_gdp) - b
        # Return the population-weighted sum of squared residuals
        return np.sum(pop * residuals**2)
    
    # Minimize the objective function with respect to a
    result = minimize_scalar(objective)
    a_best = result.x
    # Compute b using the optimal a
    b_best = np.log(np.sum(pop * per_capita_energy)) - np.log(np.sum(pop * per_capita_gdp**a_best))
    
    # Now compute the weighted R^2 on the log-transformed per_capita_energy data
    y = np.log(per_capita_energy)
    y_pred = a_best * np.log(per_capita_gdp) + b_best
    # Weighted mean of the observed log per_capita_energy values
    y_bar = np.sum(pop * y) / np.sum(pop)
    # Weighted total sum of squares and weighted residual sum of squares
    SS_tot = np.sum(pop * (y - y_bar)**2)
    SS_res = np.sum(pop * (y - y_pred)**2)
    R2 = 1 - SS_res / SS_tot
    
    return a_best, b_best, R2


#%% Main code to compute energy distribution analysis

def run_energy_dist(excel_input_data, gamma, n_percentile_levels, n_energy_levels, n_bins_out, verbose_level,  epsilon, run_name, dir, date_stamp):

    """
    Do the main energy distribution analysis for the given input data and gamma value (i.e., elastcity of energy use).

    n_percentile_levels and n_energy_levels are numbers of buckets to use in the numerical methods. These should be as high as possible
    such that decreasing a bit does not change the numbers to the precision that will be displayed.

    Try numbers like 10,000 or 1000.

    epsilon is a small number to avoid division by zero. It should be a small number like 1e-12.

    verbose_level is an integer that controls the amount of output. 0 is no output, 1 is some output, 2 is more output.

    run_name is a string that will be used to name the output files.

    dir is the parent directory of where the output files will be written.

    date_stamp is a string that will be used to name the output files and output subdirectory.
    """
    # Set the working directory to the project folder
    os.chdir(dir)

    # Create percentile vector
    pct_dx = 1. / n_percentile_levels
    percentile_vector = np.arange(0, 1 + pct_dx, pct_dx)
    percentile_vector[0] = epsilon
    percentile_vector[-1] = 1.0 - epsilon
      
    # Create energy level vector
    energy_level_vector = np.logspace(-2, 3, num=n_energy_levels)

    n_groups = 5
    group_names = ["low", "low-middle", "middle", "middle-high", "high"]
    idx_group = 6 # 6 means energy, 5 means income

    filename_prefix = f"{run_name}_p{n_percentile_levels}_e{n_energy_levels}_g{gamma:.3f}_{date_stamp}"
    
    # make directory for output files
    if not os.path.exists(filename_prefix):
        os.makedirs(filename_prefix)

    # Start timing
    start_time = time.time()

    # Step 1: Prepare country-level data

    country_summary_data = prep_country_level_lorenz_data(excel_input_data, gamma)

    key_variables = compute_key_country_level_parameters(country_summary_data, percentile_vector, energy_level_vector, verbose_level)
    elapsed_time = time.time() - start_time

    # Print the timing result
    print(f"Execution time: {elapsed_time:.2f} seconds")

    # Export country level data
    export_country_summary_data(key_variables["country_summary_data"], filename_prefix, verbose_level)

    # Export per capita and cumulative energy use by percentile for each country
    export_countries_percentile(excel_input_data, percentile_vector, key_variables["per_capita_energy_in_country_to_pop_percentile"], key_variables["cum_energy_in_country_to_pop_percentile"], filename_prefix, verbose_level)

    # Identify country groups based on per capita income
    country_groups = find_country_groups_per_capita(excel_input_data, n_groups, idx_group)
    group_indices = find_country_group_indices(country_groups, key_variables["country_summary_data"])

    # export country group information
    export_country_group_table(excel_input_data, country_groups, group_names, filename_prefix, verbose_level)

    # Export per capita and cumulative energy use by percentile and energy level for each group
    compute_percap_energy_and_cum_energy_for_groups(group_indices, group_names, "energy", excel_input_data, 
                    percentile_vector,energy_level_vector,
                    key_variables["per_capita_energy_in_country_to_pop_percentile"], 
                    key_variables["cum_energy_in_country_to_pop_percentile"], 
                    filename_prefix, verbose_level)
    
    # Combine energy data and produce aggregated results based on population and energy distribution
    combined_data = combine_energy_data(key_variables["country_summary_data"], 
                        key_variables["cum_energy_in_country_to_pop_percentile"],
                        key_variables["per_capita_energy_in_country_to_pop_percentile"], 
                        n_bins_out)
    # write out combined data
    export_combined_energy_data(combined_data, filename_prefix)

    # do energy addition and redistribution calculations
    redist = redistribute(combined_data)

    # write out redistribution data
    export_redist_energy_data(redist, filename_prefix)


#%% prepare country level data, including income distribution interpolation and gini indices

def prep_country_level_lorenz_data(excel_input_data, gamma):
    """
    Prepares country-level data by computing parameters like gamma, Lorenz curve fits,
    and Gini coefficients for income and energy distribution.

    Parameters:
        excel_input_data: pandas.DataFrame
            Input data containing country-level information. Assumes the first row contains labels,
            the second row contains global parameters, and subsequent rows contain country data.
        gamma: float
            Elasticity parameter for energy use.
        verbose_level: int
            Level of verbosity for print statements.

    Returns:
        pd.DataFrame
            A DataFrame summarizing country-level information.
    """

    cum_pop_levels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    # Step 2: Compute country-level parameters for fits to income Lorenz curve
    # The first element for each country is the spline curve, and the second element is the list of parameters
    # in the list of parameters, the first four elements are the parameters for the Jantzen-Volpert function, with 
    # the first two for the 0 to 20% range and the next two for the 80 to 100% range

    #   spline_coeffs = [coef for segment in spline.c.T for coef in segment]
    #   return spline,( [p_start, q_start, p_end, q_end] + spline_coeffs )
    lorenz_interpolation_list_country = [produce_lorenz_interpolation(cum_pop_levels, np.cumsum(row[8:18])) for row in excel_input_data.to_numpy()]

    # Step 3: Get country-level population, GDP, and energy data
    pop_list = 1000.0 * excel_input_data.iloc[:, 4].to_numpy()
    gdp_list = excel_input_data.iloc[:, 5].to_numpy()
    energy_list = (10**9 / (8760 * 3600)) * excel_input_data.iloc[:, 6].to_numpy()

    x_data = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # upper bins for income distribution data

    # Step 4: Compute integral list for energy Lorenz bin
    # Because the integral of (d Lincome/dx)^gamma from 0 to 1 is not 1, 
    # we need to compute this integral so we can use it to normalize the energy Lorenz curve so Lenergy(1) = 1.
    # to do this, we first compute the integral of (d Lincome/dx)^gamma from 0 to 1 for each country multiply time 1
    #nergy_integral_list = [
    #   compute_energy_lorenz_integral(x_data, spline_fn, p_left, q_left, p_right, q_right, gamma, 1.0)
    #   for spline_fn,[ p_left, q_left, p_right, q_right,*rest] in lorenz_interpolation_list_country]

    energy_integral_list = []
    for spline_fn, values in lorenz_interpolation_list_country:
        p_left, q_left, p_right, q_right, *rest = values  # Unpack properly
        energy_integral = compute_energy_integral(0, 1, x_data, spline_fn, p_left, q_left, p_right, q_right, gamma)
        energy_integral_list.append(energy_integral)

    if verbose_level > 2:
        print("energy integral list: ", energy_integral_list)
         
    # Step 5: Compute income Gini list
    income_gini_list = [
        1 - compute_income_lorenz_integral(0.0, 1.0, x_data, spline_fn, p_left, q_left, p_right, q_right) / 0.5
        for spline_fn,[ p_left, q_left, p_right, q_right,*rest] in lorenz_interpolation_list_country]


   # Step 6: Compute energy Gini list
    energy_gini_list = [
        1 - compute_energy_lorenz_integral(x_data, spline_fn, p_left, q_left, p_right, q_right, gamma, energy_integral_list[idx]) / 0.5
        for idx,( spline_fn,[ p_left, q_left, p_right, q_right,*rest]) in enumerate( lorenz_interpolation_list_country) ]


    # Step 7: Create country summary table
    country_summary_data = {
        "Country Name": excel_input_data.iloc[:, 1].to_numpy(),
        "Country Code": excel_input_data.iloc[:, 2].to_numpy(),
        "Population": pop_list,
        "GDP": gdp_list,
        "Energy": energy_list,
        "Gamma": [gamma] * len(pop_list),
        "Energy Integral": energy_integral_list,
        "Income Gini": income_gini_list,
        "Energy Gini": energy_gini_list,
        "Pleft": [item[1][0] for item in lorenz_interpolation_list_country],  # Extract Pleft into its own list
        "Qleft": [item[1][1] for item in lorenz_interpolation_list_country],  # Extract Qleft into its own list
        "Pright": [item[1][2] for item in lorenz_interpolation_list_country],  # Extract Pright into its own list
        "Qright": [item[1][3] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_20_30_3": [item[1][4] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_20_30_2": [item[1][5] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_20_30_1": [item[1][6] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_20_30_0": [item[1][7] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_30_40_3": [item[1][8] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_30_40_2": [item[1][9] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_30_40_1": [item[1][10] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_30_40_0": [item[1][11] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_40_50_3": [item[1][12] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_40_50_2": [item[1][13] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_40_50_1": [item[1][14] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_40_50_0": [item[1][15] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_50_60_3": [item[1][16] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_50_60_2": [item[1][17] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_50_60_1": [item[1][18] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_50_60_0": [item[1][19] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_60_70_3": [item[1][20] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_60_70_2": [item[1][21] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_60_70_1": [item[1][22] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_60_70_0": [item[1][23] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_70_80_3": [item[1][24] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_70_80_2": [item[1][25] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_70_80_1": [item[1][26] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_70_80_0": [item[1][27] for item in lorenz_interpolation_list_country],  # Extract Qright into its own list
        "spline_fn": [item[0] for item in lorenz_interpolation_list_country]  # Extract spline into its own list

    }

    return country_summary_data

#%% compute values needed to compute energy and income gini indices

def produce_lorenz_interpolation(x_data, y_data):
    """
    Process x_data and y_data to fit Jantzen-Volpert function to the first and last pair of points,
    compute derivatives, and fit a cubic spline through the intermediate points.
    """
    # Fit Jantzen-Volpert function to first and last pair of points
    # fit start on 10 and 20% numbers
    p_start, q_start = find_jantzen_volpert_p_q(x_data[0], y_data[0], x_data[1], y_data[1])
    # fit end on 80 and 90% numbers
    p_end, q_end = find_jantzen_volpert_p_q(x_data[-3], y_data[-3], x_data[-2], y_data[-2])
    
    # Compute derivatives at the second (20%) and next-to-next-to-last (80%) data points
    dy_dx_start = jantzen_volpert_fn_deriv(x_data[1], p_start, q_start)
    dy_dx_end = jantzen_volpert_fn_deriv(x_data[-3], p_end, q_end)
    
    # Fit the spline using second to next-to-last points
    spline_fn = fit_monotonic_convex_spline_with_derivatives(x_data[1:-2], y_data[1:-2], dy_dx_start, dy_dx_end)


    spline_coeffs = [coef for segment in spline_fn.c.T for coef in segment]
    return spline_fn,( [p_start, q_start, p_end, q_end] + spline_coeffs )

# find jantzen volpert fit going through two points

def find_jantzen_volpert_p_q(x0, y0, x1, y1):
    # Initial guess for p and q
    initial_guess = [0.5, 0.5]
    
    # Data points
    x_data = np.array([x0, x1])
    y_data = np.array([y0, y1])
    
    # Curve fitting to determine p and q
    popt, _ = curve_fit(jantzen_volpert_fn, x_data, y_data, p0=initial_guess, bounds=((0, 0), (1, 1)))
    
    return popt[0], popt[1]  # p and q
"""
# Example usage
x0, y0 = 0.2, 0.1  # Given values for x0 and f(x0)
x1, y1 = 0.8, 0.6  # Given values for x1 and f(x1)
p, q = find_p_q(x0, y0, x1, y1)
print(f"Estimated p: {p}, Estimated q: {q}")
"""


def jantzen_volpert_fn(x, p, q):
    return x**p * (1 - (1 - x)**q)
    #   == x**p - x**p * (1 - x)**q


def jantzen_volpert_fn_deriv(x, p, q):
    term1 = p * x**(p - 1) * (1 - (1 - x)**q)
    term2 = x**p * q * (1 - x)**(q - 1)
    return term1 + term2


def fit_monotonic_convex_spline_with_derivatives(x_data, y_data, dy_dx_start, dy_dx_end):
    """
    Fits a cubic spline ensuring:
    - Monotonic increasing function
    - First derivative is continuous and increasing
    - Uses given first derivatives at the endpoints
    """
    # Fit a cubic spline with first derivative constraints at the endpoints
    spline_fn = CubicSpline(x_data, y_data, bc_type=((1, dy_dx_start), (1, dy_dx_end)), extrapolate=False)
    
    return spline_fn


def compute_energy_integral(x_left,x_right,x_data, spline_fn, p_left, q_left, p_right, q_right, gamma):
    """
    Integrates (d f(x)/d x)^gamma over [x_left, x_right] where f(x) is defined piecewise:
      - For x in [0, x_data[1]]: f(x) = jantzen_volpert_fn(x, p_left, q_left)
      - For x in [x_data[1], x_data[-3]: f(x) = spline_fn(x)
      - For x in [x_data[-3, 1]: f(x) = jantzen_volpert_fn(x, p_right, q_right)
      
    Parameters:
        x_data : array-like
            Array of x-values spanning [0, 1] (with x_data[0] == 0 and x_data[-1] == 1).
        spline_fn : CubicSpline
            The fitted spline function for the middle segment.
        p_left, q_left : float
            Parameters for the analytic function on the left segment.
        p_right, q_right : float
            Parameters for the analytic function on the right segment.
        gamma : float
            The exponent to which f(x) is raised.
            
    Returns:
        total_integral : float 
            The numerical value of the integral of f(x)^gamma from 0 to 1.
    """
    # Integrate the left analytic segment from 0 to x_data[1]:
    if x_left < x_data[1]: # i.e., if x_right < x_data[1] (i.e., 20%)
        integral_left, _ = quad(lambda x: jantzen_volpert_fn_deriv(x, p_left, q_left)**gamma,
                            x_left, min(x_right,x_data[1]))
    else:
        integral_left = 0.0

    # Integrate the middle spline segment numerically from x_data[1] to x_data[-3]: 20% to 80%
    if x_left < x_data[-3] and x_right > x_data[1]:
        spline_derivative = spline_fn.derivative()
        integral_middle, error = quad(lambda x: spline_derivative(x)**gamma, max(x_left,x_data[1]), min(x_right,x_data[-3]))
    else:
        integral_middle = 0.0


    # Integrate the right analytic segment from x_data[-] to 1:
    if x_right > x_data[-3]: # i.e., if x_right > x_data[-2] (i.e., 80%)
        integral_right, _ = quad(lambda x: jantzen_volpert_fn_deriv(x, p_right, q_right)**gamma,
                             max(x_left,x_data[-3]),x_right)
    else:
        integral_right = 0.0
    
    # Sum the three pieces to get the total integral:
    total_integral = integral_left + integral_middle + integral_right
    #print("integral_left, integral_middle, integral_right, total_integral: ", integral_left, integral_middle, integral_right, total_integral)
    #print("energy_integral: ", total_integral)
    return total_integral 

def compute_energy_lorenz_integral(x_data, spline_fn, p_left, q_left, p_right, q_right, gamma, energy_integral):
    """
    Integrates (d f(x)/d x)^gamma over [0, 1] where f(x) is defined piecewise:
      - For x in [0, x_data[1]]: f(x) = jantzen_volpert_fn(x, p_left, q_left)
      - For x in [x_data[1], x_data[-3]: f(x) = spline_fn(x)
      - For x in [x_data[-3, 1]: f(x) = jantzen_volpert_fn(x, p_right, q_right)
      
    Parameters:
        x_data : array-like
            Array of x-values spanning [0, 1] (with x_data[0] == 0 and x_data[-1] == 1).
        spline_fn : CubicSpline
            The fitted spline function for the middle segment.
        p_left, q_left : float
            Parameters for the analytic function on the left segment.
        p_right, q_right : float
            Parameters for the analytic function on the right segment.
        gamma : float
            The exponent to which f(x) is raised.
            
    Returns:
        total_integral : float 
            The numerical value of the integral of f(x)^gamma from 0 to 1.
    """
    # Integrate the left analytic segment from 0 to x_data[1]:
    total_integral, _ = quad(lambda x:  compute_energy_integral(0,x,x_data, spline_fn, p_left, q_left, p_right, q_right, gamma),
                            0., 1.)
    return total_integral / energy_integral




def compute_income_lorenz_integral(x0,x1,x_data, spline_fn, p_left, q_left, p_right, q_right):
    # Integrate the spline function with jantzen volpert ends from x0 to x1
    # assumeing x1 > x0

    # x_data are the x data values for the upper end of the income bins

    integral = 0.0
    
    # Integrate the left analytic segment from 0 to x_data[1].
    if x0 <= x_data[1]: # i.e., if x0 < x_data[1] (i.e., 20%)
        integral_left = jantzen_volpert_fn_integral(x0, min(x1,x_data[1]), p_left, q_left)
    else:
        integral_left = 0.0

    
    # Integrate the middle spline segment using its antiderivative.
    if x0 <= x_data[-3] and x1 > x_data[1]:
        integral_middle = spline_fn.integrate(max(x0,x_data[1]),min( x1,x_data[-3])) # i.e., max(x0, 20%) to min(x1, 80%)
    else:
        integral_middle = 0.0
    
    # Integrate the right analytic segment from x_data[-2] to 1.
    if x1 >= x_data[-3]: # i.e., if x1 > x_data[-2] (i.e., 80%)
        integral_right = jantzen_volpert_fn_integral(max(x0,x_data[-3]), x1, p_right, q_right)
    else:
        integral_right = 0.0
    
    # Sum the three pieces to get the total integral.
    total_integral = integral_left + integral_middle + integral_right

    return total_integral


def jantzen_volpert_fn_integral(x0, x1, p, q):
    """
    Computes:
      ((-x0^(1+p) + x1^(1+p))/(1+p) + Beta[x0, 1+p, 1+q] - Beta[x1, 1+p, 1+q])
    with the condition that x1 < 1 and x0 > 0.
    
    Here, Beta[x, a, b] is the (lower) incomplete beta function:
      Beta[x, a, b] = int_0^x t^(a-1) * (1-t)^(b-1) dt.
      
    In SciPy, betainc(a, b, x) returns the regularized incomplete beta function I_x(a,b),
    so we multiply by beta(a, b) to get the unregularized value.
    """
    # Compute the first term
    term1 = (-x0**(1+p) + x1**(1+p)) / (1+p)

    # Compute the unregularized incomplete beta functions for x0 and x1.
    # Note: In SciPy, the lower incomplete beta is computed as:
    #       B_x(a,b) = betainc(a, b, x) * beta(a, b)
    a = 1 + p
    b = 1 + q
    incomplete_beta_x0 = betainc(a, b, x0) * beta_func(a, b)
    incomplete_beta_x1 = betainc(a, b, x1) * beta_func(a, b)

    return term1 + incomplete_beta_x0 - incomplete_beta_x1


#%% Create key country level intemediate values by population percentile needed to aggregate across countries

def compute_key_country_level_parameters(country_summary_data, percentile_vector, energy_level_vector,  verbose_level):
    """
    Processes input data to compute various energy-related tables by country and percentile of population.
    
    Parameters:
        excel_input_data: str
            Path to the Excel file containing country-level data.
        percentile_vector: list
            List of percentiles (e.g., np.arange(0, 1.01, 0.01)).
        n_energy_levels: int
            Number of energy levels to compute.
        verbose_level: int
            Level of verbosity for print statements.
    Returns:
        dict
            A dictionary containing calculated tables and lists.
    """

    if verbose_level > 0:
        print(f"Generating list by country and percentile of population; {datetime.now()}")

    # Step 2: Generate per capita energy boundaries by country
    per_capita_energy_in_country_to_pop_percentile, cum_energy_in_country_to_pop_percentile = gen_country_summary_data_by_fract_of_pop(
        country_summary_data, percentile_vector,epsilon, verbose_level
    )

    # Step 4: Compute fraction of population by country and energy level
    if verbose_level > 0:
        print(f"Computing fractPopTable; {datetime.now()}")
    fract_pop_table = fract_pop_in_country_to_energy_per_capita_level(
        per_capita_energy_in_country_to_pop_percentile, percentile_vector, energy_level_vector
    )

    # Step 5: Compute population by country and energy level
    pop_table = np.array(country_summary_data["Population"])[:, np.newaxis] * fract_pop_table

    # Return the results in a dictionary
    return {
        "country_summary_data":country_summary_data,
        "per_capita_energy_in_country_to_pop_percentile": per_capita_energy_in_country_to_pop_percentile,
        "cum_energy_in_country_to_pop_percentile": cum_energy_in_country_to_pop_percentile,
        "energy_level_vector": energy_level_vector,
        "fract_pop_table": fract_pop_table,
        "pop_table": pop_table,
        "cum_energy_in_country_to_pop_percentile": cum_energy_in_country_to_pop_percentile
    }


def gen_country_summary_data_by_fract_of_pop(country_summary_data, percentile_vector, epsilon, verbose_level):
    """
    Generates lists of per capita energy use and cumulative energy use
    by population percentile for each country.

    Parameters:
        country_summary_data: dict
            The dictionary containing country-level data.
        percentile_vector: list
            A list of percentiles (e.g., np.arange(0, 1.01, 0.01)).
        filename_prefix: str
            Name to be used in output (not used directly here but passed).
        verbose_level: int
            Level of verbosity for print statements.

    
            - per_capita_energy_in_country_to_pop_percentile: numpy array of per capita energy use.
            - cum_energy_in_country_to_pop_percentile: numpy of cumulative energy use.
    """
    if verbose_level > 0:
        print("Generating lists of per capita and cumulative energy use by population percentile...")

    x_data = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # upper bounds of income distribution categoris
    n_countries = len(country_summary_data["Population"])
    n_percentiles = len(percentile_vector)


    # Generate cumulative energy use by population percentile
    # integrate_energy_in_pop_pctile_range(x0, x1, x_data, spline, 
    # #        p_left, q_left, p_right, q_right, gamma, energy, energy_integral):

    cum_energy_in_country_to_pop_percentile = []

    for idx_country in range(n_countries):
        if verbose_level > 1:
            print(f"Processing country index: {idx_country}")  # Your print statement

        country_cumulative_energy = np.cumsum([
            integrate_energy_in_pop_pctile_range(
                percentile_vector[max(0, i - 1)],
                percentile_vector[i],
                x_data,
                country_summary_data["spline_fn"][idx_country], 
                country_summary_data["Pleft"][idx_country], 
                country_summary_data["Qleft"][idx_country],
                country_summary_data["Pright"][idx_country], 
                country_summary_data["Qright"][idx_country], 
                country_summary_data["Gamma"][idx_country], 
                country_summary_data["Energy"][idx_country],
                country_summary_data["Energy Integral"][idx_country]
            )
            for i in range(len(percentile_vector))
        ])

        cum_energy_in_country_to_pop_percentile.append(country_cumulative_energy)

    cum_energy_in_country_to_pop_percentile = np.array(cum_energy_in_country_to_pop_percentile)

    # calculate per capita energy use in country by pop percentile bin by calculating the amount of energy in
    # each percentile bin divided by the population in that percentile bin
    # The amount in the bin is the difference between the cumulative energy in the bin and the cumulative energy in the previous bin
    # The population in the bin is the difference between the cumulative population in the bin and the cumulative population in the previous bin
    energy_in_bin_by_country = cum_energy_in_country_to_pop_percentile[:,1:] - cum_energy_in_country_to_pop_percentile[:,:-1]
    
    pct_deltas = (percentile_vector[1:] - percentile_vector[:-1])[np.newaxis,:]  # shape (C, P)
    pop = country_summary_data["Population"][:, np.newaxis]  # shape (C, P)
    pop_in_bin_by_country = pop * pct_deltas  # shape (C, P)

    per_capita_energy_in_bin = energy_in_bin_by_country / pop_in_bin_by_country

    # Initialize the array before using it
    per_capita_energy_in_country_to_pop_percentile = np.zeros((n_countries, n_percentiles))

    # now for each country we need to interpolate that to the original percentile list
    # for the first and last elements of the percentile list, we will linearly extrapolate from the 
    # two closes values   # make a variable that is the same shape as cum_energy_in_country_to_pop_percentile
    midpoints = (percentile_vector[1:] + percentile_vector[:-1]) / 2
    for idx_country in range(n_countries):
        f = interp1d(midpoints, per_capita_energy_in_bin[idx_country], fill_value="extrapolate")
        per_capita_energy_in_country_to_pop_percentile[idx_country] = f(percentile_vector)

  
    if verbose_level > 0:
        print("Completed generating energy use lists.")

    return per_capita_energy_in_country_to_pop_percentile, cum_energy_in_country_to_pop_percentile

def energy_per_capita_fn(x, x_data, spline_fn, p_left, q_left, p_right, q_right, gamma, energy, pop, energy_integral):
    """
    Computes the energy per capita usage levels for a country based on the given parameters.

    Parameters:
        x: float
            A scalar value, must be numeric.
        x_data: list of upper bound of income levels
        spline_fn: result of CubicSpline fit (for middle of income lorenz curve)
        p_left, q_left: floats
            Parametersfor the left segment of the Lorenz curve.
        gamma: float
            Elasticity parameter.
        pop: float
            Country level population.
        energy: float
            Total energy use by country

    Returns:
        float
            The energy per capita value at that percentile value.
    """

    # Compute the derivative of energy use with respect to x (population percentile)
    deriv = compute_d_energy_lorenz_dx(x, x_data, spline_fn, p_left, q_left, p_right, q_right, gamma, energy_integral)

    # return derivative times mean country level energy use
    return deriv * energy / pop

def compute_d_energy_lorenz_dx(x, x_data, spline_fn, p_left, q_left, p_right, q_right, gamma, energy_integral):
    """
    Computes f"(x)^gamma with respect to x for 0 <= x <= 1, where f(x) is defined piecewise as:
      - For x in [0, x_data[1]]:
            f(x) = jantzen_volpert_fn(x, p_left, q_left)
      - For x in [x_data[1], x_data[-2]]:
            f(x) = spline_fn(x)
      - For x in [x_data[-2], 1]:
            f(x) = jantzen_volpert_fn(x, p_right, q_right)
    
    The derivative is computed as:
    
        d/dx (f(x)^gamma) = gamma * f(x)^(gamma-1) * f'(x)
    
    Parameters:
        x : float
            The point at which to evaluate the derivative.
        x_data : array-like
            Array of x-values spanning [0, 1].
        spline_fn : CubicSpline
            The fitted spline function for the middle segment.
        p_left, q_left : float
            Parameters for the analytic function on the left segment.
        p_right, q_right : float
            Parameters for the analytic function on the right segment.
        gamma : float
            The exponent applied to f(x).
    
    Returns:
        float
            The derivative of f(x)^gamma at the point x.
    """
    # Compute f(x) and f'(x) based on which segment x falls into.
    if x <= x_data[1]:
        f_prime = jantzen_volpert_fn_deriv(x, p_left, q_left)
    elif x <= x_data[-3]:
        f_prime = spline_fn.derivative()(x)
    else:
        f_prime = jantzen_volpert_fn_deriv(x, p_right, q_right)
    
    return  f_prime**gamma / energy_integral


def compute_d_income_lorenz_dx(x, x_data, spline_fn, p_left, q_left, p_right, q_right):
    """
    Computes the derivative of the piecewise function f(x) at a given x, where f(x)
    is defined as:
      - For x in [0, x_data[1]]:
            f(x) = jantzen_volpert_fn(x, p_left, q_left)
      - For x in [x_data[1], x_data[-2]]:
            f(x) = spline_fn(x)
      - For x in [x_data[-2], 1]:
            f(x) = jantzen_volpert_fn(x, p_right, q_right)
            
    Parameters:
        x : float
            The point at which the derivative is evaluated.
        x_data : array-like
            Array of x-values spanning [0, 1] (with x_data[0] == 0 and x_data[-1] == 1).
        spline_fn : CubicSpline
            The fitted spline function for the middle segment.
        p_left, q_left : float
            Parameters for the analytic function on the left segment.
        p_right, q_right : float
            Parameters for the analytic function on the right segment.
            
    Returns:
        float
            The derivative f'(x) evaluated at the given x.
    """
    # Left analytic segment: 0 <= x <= x_data[1]
    if x <= x_data[1]:
        return jantzen_volpert_fn_deriv(x, p_left, q_left)
    
    # Middle spline segment: x_data[1] < x < x_data[-2]
    elif x <= x_data[-3]:
        # Compute the derivative using the spline's derivative.
        return spline_fn.derivative()(x)
    
    # Right analytic segment: x_data[-2] <= x <= 1
    else:
        return jantzen_volpert_fn_deriv(x, p_right, q_right)
    
def integrate_energy_in_pop_pctile_range(x0, x1, x_data, spline_fn, p_left, q_left, p_right, q_right, gamma, energy, energy_integral):
    """
    Integrates compute_d_energy_lorenz_dx from x0 to x1.
    
    Parameters:
        x0, x1 : float
            The integration bounds.
        x_data, spline_fn, p_left, q_left, p_right, q_right, gamma, energy_integral : parameters
            Same as those used in compute_d_energy_lorenz_dx.
    
    Returns:
        float
            The integral of d(energy_lorenz)/dx from x0 to x1.
    """
    #print ("x0, x1", x0, x1,
    #       "p_left, q_left, p_right, q_right, gamma, energy, energy_integral", 
    #       p_left, q_left, p_right, q_right, gamma, energy, energy_integral)
    def integrand(x):
        return compute_d_energy_lorenz_dx(x, x_data, spline_fn, p_left, q_left, p_right, q_right, gamma, energy_integral)
    
    result, error = quad(integrand, x0, x1)
    return energy * result

def slope_estimate(x_data, y_data):
    """
    Estimates the slope of a line through the data points (x_data, y_data).
    
    Parameters:
        x_data: list of floats
            The x-coordinates of the data points.
        y_data: list of floats
            The y-coordinates of the data points.

    Returns:
        float
            The estimated slope of the line through the data points.
    """
    # Calculate the slope using the finite difference method
    slopes = np.diff(y_data) / np.diff(x_data)
    midpoints = (x_data[1:] + x_data[:-1]) / 2
    
    # slopes is now a good estimate of the slope at the midpoints of the data points
    # we need to now interpolate the slope to the x_data points
    f = interp1d(midpoints, slopes, fill_value="extrapolate")
    return f(x_data)        

 def slope_log_estimate(x_data, y_data):
    """
    Estimates the slope of a line through the data points (x_data, y_data).
    Sometimes we want the result to be in the form of how much y changes per percent increase in x

    This is when slope_log_estimate is called.  The idea is that this is appropriate for functions that are linear in the log of x.
    
    Parameters:
        x_data: list of floats
            The x-coordinates of the data points.
        y_data: list of floats
            The y-coordinates of the data points.

    Returns:
        float
            The estimated slope of the line through the data points.
    """
    # Calculate the slope using the finite difference method
    slopes = np.diff(y_data) / np.diff(np.log(x_data))
    logMidpoints = (np.log(x_data[1:]) + np.log(x_data[:-1])) / 2  
    # This is taking the geometric mean which is consistent with a log-linear relationship

    # slopes is now a good estimate of the slope at the midpoints of the data points
    # we need to now interpolate the slope to the x_data points
    f = interp1d(logMidpoints, slopes, fill_value="extrapolate")
    return f(x_data)        

    



#%% compute values as a function of per capita energy level

def fract_pop_in_country_to_energy_per_capita_level(per_capita_energy_in_country_to_pop_percentile, percentile_vector, energy_level_vector):
    """
    Computes the fraction of the population by energy per capita level.

    Parameters:
        per_capita_energy_in_country_to_pop_percentile: np.array
            A list where each element corresponds to a country's per capita energy boundaries.
        percentile_vector: list
            A list of percentiles.
        energy_level_vector: list
            A list of energy values to map to percentiles.

    Returns:
        np.array
            A list where each element corresponds to a country's fraction of population mapped to energy levels.
    """
    result = []
    for idx in range(len(per_capita_energy_in_country_to_pop_percentile)):
        # Extract per capita energy boundaries for the current country
        per_capita_energy = per_capita_energy_in_country_to_pop_percentile[idx]
        
        # Determine the minimum and maximum energy levels
        min_energy = min(per_capita_energy)
        max_energy = max(per_capita_energy)
        
        # Create a linear interpolation function
        interp_fn = interp1d(per_capita_energy, percentile_vector, bounds_error=False, fill_value=(percentile_vector[0], percentile_vector[-1]))
        
        # Map energy values to percentiles, clamping them to the [min_energy, max_energy] range
        mapped_values = [interp_fn(np.clip(e_val, min_energy, max_energy)) for e_val in energy_level_vector]
        
        # Append the mapped values for the current country
        result.append(mapped_values)
    
    return np.array(result)

#-------------------------------------------------------------------------------------------------------------


def energy_in_country_to_fract_pop_level(cum_energy_in_country_to_pop_percentile, percentile_vector, pop_fract_list):
    """
    Maps energy use to specific population percentiles for each country.

    Parameters:
        cum_energy_in_country_to_pop_percentile: list of np.array
            Each array contains the cumulative energy values for a country, ordered by population percentiles.
        percentile_vector: list
            List of percentiles (e.g., [0.0, 0.1, ..., 1.0]) that define the x-values for interpolation.
        pop_fract_list: list of lists
            Each sublist contains population fraction levels (percentiles) for a specific country.

    Returns:
        cum_energy_in_country_to_pop_percentile: np.array
            A 2D array where each row corresponds to a country, and each column corresponds to 
            the energy use at the respective population percentile.
    """
    cum_energy_in_country_to_pop_percentile = []

    for idx in range(len(cum_energy_in_country_to_pop_percentile)):
        cum_energy = cum_energy_in_country_to_pop_percentile[idx]

        # Create interpolation function from population percentiles to energy values
        interp_fn = interp1d(percentile_vector, cum_energy, bounds_error=False, fill_value=(cum_energy[0], cum_energy[-1]))

        # Clamp pop fractions to valid range of interpolation, then map to energy values
        clamped_fracts = np.clip(pop_fract_list[idx], percentile_vector[0], percentile_vector[-1])
        mapped_values = interp_fn(clamped_fracts)

        cum_energy_in_country_to_pop_percentile.append(mapped_values)

    return np.array(cum_energy_in_country_to_pop_percentile)


#%%
# Code to summarize key variables and output as xlsx files

def find_country_groups_per_capita(excel_input_data, n_groups, idx_group):
    """
    Groups countries by per capita income or energy use.

    Parameters:
        excel_input_data: pandas.DataFrame
            Input data where rows correspond to countries, and columns include population, income, and energy data.
        n_groups: int
            Number of groups to create.
        idx_group: int
            Index for the column to group by (5 for income or 6 for energy).

    Returns:
        list of lists:
            A list where each element corresponds to a group's countries.
    """
    # Step 1: Sort data by the chosen per capita metric
    excel_input_data["PerCapita"] = excel_input_data.iloc[:, idx_group] / excel_input_data.iloc[:, 4]  # Per capita value (e.g., income/pop)
    print (excel_input_data["PerCapita"], excel_input_data.iloc[:, idx_group], excel_input_data.iloc[:, 4])
    sorted_data = excel_input_data.sort_values(by="PerCapita").reset_index(drop=True)
    print (sorted_data)

    # Step 2: Calculate cumulative population, income, and energy
    cum_pop = sorted_data.iloc[:, 4].cumsum()
    cum_pop /= cum_pop.iloc[-1]  # Normalize to [0, 1]

    cum_income = sorted_data.iloc[:, 5].cumsum()
    cum_income /= cum_income.iloc[-1]  # Normalize to [0, 1]

    cum_energy = sorted_data.iloc[:, 6].cumsum()
    cum_energy /= cum_energy.iloc[-1]  # Normalize to [0, 1]

    print (cum_pop, cum_income, cum_energy)

    # Step 3: Determine country positions corresponding to population groups
    pop_targets = np.linspace(1. / n_groups, 1, n_groups)
    #country_positions = [np.searchsorted(cum_pop, target-epsilon) for target in pop_targets]
    country_positions = np.array([np.searchsorted(cum_pop, target) for target in pop_targets])
    # Convert to NumPy array to allow for NumPy-style indexing
  

    # check for duplicates in country_positions
    duplicates = np.where(np.diff(country_positions) == 0)[0]
    if duplicates.size > 0:
        print ("Duplicates found in country_positions: ", duplicates)
        print ("subtracting 1 from the duplicates as kluge")
        country_positions[duplicates] -= 1

    print ("pop_targets ", pop_targets)
    print ("country_positions ", country_positions)

    # Step 4: Create the table with relevant data
    country_pos_data = sorted_data.iloc[country_positions]
    tab = pd.DataFrame({
        "Country": country_pos_data.iloc[:, 1],  # Country names
        "PerCapita": country_pos_data.iloc[:, idx_group] / (1000 * country_pos_data.iloc[:, 4]),  # Per capita
        "CumPop": cum_pop.iloc[country_positions].to_numpy(),
        "CumIncome": cum_income.iloc[country_positions].to_numpy(),
        "CumEnergy": cum_energy.iloc[country_positions].to_numpy()
    })

    # Step 5: Compute differences for cumulative columns
    tab["CumPop"] = tab["CumPop"].diff().fillna(tab["CumPop"].iloc[0])
    tab["CumIncome"] = tab["CumIncome"].diff().fillna(tab["CumIncome"].iloc[0])
    tab["CumEnergy"] = tab["CumEnergy"].diff().fillna(tab["CumEnergy"].iloc[0])

    # Print the table if needed
    print(tab)

    # Step 6: Select countries for each group based on the tabulated per capita range
    groups = []
    for idx in range(n_groups):
        lower_bound = 1000 * tab["PerCapita"].iloc[idx - 1] if idx > 0 else 0
        upper_bound = 1000 * tab["PerCapita"].iloc[idx]
        group = sorted_data[
            (sorted_data.iloc[:, idx_group] / sorted_data.iloc[:, 4] > lower_bound) &
            (sorted_data.iloc[:, idx_group] / sorted_data.iloc[:, 4] <= upper_bound)
        ].iloc[:, 2].to_list()  # Extract column 3 (country codes)
        groups.append(group)

    return groups

#-------------------------------------------------------------------------------------------------------------

def find_country_group_indices(country_groups, country_summary_data):
    """
    Finds the indices of countries in the country list that belong to each group.

    Parameters:
        country_groups: list of lists
            A list where each sublist contains country identifiers (e.g., country codes) for a specific group.
        country_summary_data: dict
            A dictionary containing country-level data, where "Country Code" is a key referencing country identifiers.

    Returns:
        list of lists:
            A list where each sublist contains the indices of countries in the `country_summary_data` that belong to the respective group.
    """
    country_codes = country_summary_data["Country Code"]  # Extract country codes from the country list

    # Find indices for each group
    group_indices = [
        [idx for idx, code in enumerate(country_codes) if code in group]
        for group in country_groups
    ]

    return group_indices



def export_country_group_table(excel_input_data, country_groups, group_names, filename_prefix, verbose_level):
    """
    Exports a table of country names, country codes, populations, per-capita GDP, per-capita income, and group membership.

    Parameters:
        excel_input_data: pandas.DataFrame
            Input data containing country-level information.
        country_groups: list of lists
            Indices of countries in each group.
        group_names: list
            Names of the groups.
        filename_prefix: str
            Name used in the output file name.
        verbose_level: int
            Level of verbosity for printing progress and results.

    Returns:
        None
    """
    # Create a list to store the output data
    output_data = []

    # Iterate over each group and its corresponding name
    print ("country_groups", country_groups)
    for group_idx, group in enumerate(country_groups):
        print (group_idx,group)
        group_name = group_names[group_idx]
        for country_code in group:
            # Find the index of the country in the input data
            country_idx = excel_input_data[excel_input_data.iloc[:, 2] == country_code].index[0]
            country_name = excel_input_data.iloc[country_idx, 1]
            country_code = excel_input_data.iloc[country_idx, 2]
            population = excel_input_data.iloc[country_idx, 4]
            per_capita_gdp = excel_input_data.iloc[country_idx, 5] / population
            per_capita_income = excel_input_data.iloc[country_idx, 6] / population
            output_data.append([country_name, country_code, population, per_capita_gdp, per_capita_income, group_name])

    # Define column headings
    headings = ["Country Name", "Country Code", "Population", "Per Capita GDP", "Per Capita Income", "Group"]

    # Create a pandas DataFrame
    df = pd.DataFrame(output_data, columns=headings)

    # Create the output file name
    file_name = f"./{filename_prefix}/{filename_prefix}_country_group_table.xlsx"

    # Export to excel
    df.to_excel(file_name, index=False)

    if verbose_level > 0:
        print(f"Exported {file_name}")

#-------------------------------------------------------------------------------------------------------------

def prepend_zero_column(data):
    """
    Prepends a zero column to the input data.

    Parameters:
        np.array
            The input data.

    Returns:
        np.array
            The input data with a zero column prepended.
    """
    return np.column_stack((np.zeros((data.shape[0], 1)), data))
    
#-------------------------------------------------------------------------------------------------------------
def compute_global_energy_lorenz_curve_and_derivative(excel_input_data, cum_energy_in_country_to_pop_percentile, 
                 per_capita_energy_in_country_to_pop_percentile, percentile_vector, filename_prefix, verbose_level):

    """
    Computes the global energy Lorenz curve and its derivative.

    Parameters:
        excel_input_data: pandas.DataFrame
            Data containing country names, codes, and energy information.
        filename_prefix: str
            Name used in the output file names.
        verbose_level: int
            Level of verbosity for printing progress and results.

    Returns:    
        global_energy_lorenz_curve: list of floats
            The cumulative energy use by population percentile.
        global_energy_lorenz_derivative: list of floats
            The derivative of the cumulative energy use by population percentile.
    """
    pop_scalar = np.sum(excel_input_data.iloc[:, 4] )
    # make an array for each country in the group showing the population in each percentile bin
    pop_countries_vector = excel_input_data.iloc[idx, 4] 
    # calculate percentile bin widths, under the assumption that there is a tacit zero for the lower bound of the first bin
    bin_widths = percentile_vector[1:] - percentile_vector[:-1]
    # make an array for each country in the group showing the population in each percentile bin
    pop_country_pct_array = pop_countries_vector[:, np.newaxis] * np.concatenate(([0],bin_widths))




    # flatten the arrays
    energy_in_bin_array = per_capita_energy_in_country_to_pop_percentile
    per_capita_energy_vector = np.flatten(per_capita_energy_in_country_to_pop_percentile)
    population vector

    # now calculate the amount of energy use in each bin, by subtracting the cumulative energy use 
    # at the lower bound of the bin from the cumulative energy use at the upper bound of the bin,
    # assuming an implict zero for the lower bound of the first bin
    # Assuming your 2D array is called 'array_2d'
    energy_array = np.diff(cum_energy_array)
    zeros_column = np.zeros((energy_array.shape[0], 1))  # Create a column of zeros with same number of rows
    energy_array = np.column_stack((zeros_column, energy_array))    


    

#-------------------------------------------------------------------------------------------------------------

def compute_percap_energy_and_cum_energy_for_groups(groups, group_names, group_type, excel_input_data, 
                      percentile_vector, energy_level_vector, per_capita_energy_in_country_to_pop_percentile, 
                      cum_energy_in_country_to_pop_percentile, filename_prefix, verbose_level):
    """
    The goal of this function is to export the population and cumulative energy use by percentile for each group.

    The output is a table with the group name, and the population and cumulative energy use by percentile for each group.

    The output is two tables, one for per capita energy use by percentile of population in the group,
    and the other is for the cumulative  energy use by percentile of population in the group.

    Parameters:
        groups: list of lists
            Indices of countries in each group.
        group_names: list
            Names of the groups.
        group_type: str
            Type of the group, used in file names.
        excel_input_data: pandas.DataFrame
            Data containing population and energy information for countries.
        percentile_vector: list
            List of percentiles.
        per_capita_energy_in_country_to_pop_percentile: list of lists
            Per capita energy use by population percentile for each country.
        cum_energy_in_country_to_pop_percentile: list of lists
            Cumulative energy use by population percentile for each country.
        filename_prefix: str
            Name used in the output file names.
        verbose_level: int
            Level of verbosity for printing progress and results.

    Returns:
        None
    """
    group_percap_energy_pct_interp = []
    group_cum_energy_pct_interp = []
    group_cum_pop_percap_energy_interp = []
    group_cum_energy_percap_energy_interp = []

    # Loop over each group to make the logic clear
    for group_idx, group in enumerate(groups):
        # calculate the total population in the group
        pop_group_scalar = np.sum([excel_input_data.iloc[idx, 4] for idx in group])
        # make an array for each country in the group showing the population in each percentile bin
        pop_countries_vector = np.array([excel_input_data.iloc[idx, 4] for idx in group])
        # calculate percentile bin widths, under the assumption that there is a tacit zero for the lower bound of the first bin
        bin_widths = percentile_vector[1:] - percentile_vector[:-1]
        # make an array for each country in the group showing the population in each percentile bin
        pop_bin = pop_countries_vector[:, np.newaxis] * np.concatenate(([0],bin_widths))
        # now make an array for each country in the group showing the per capita energy use at each bin boundary
        pc_energy_array = np.array([per_capita_energy_in_country_to_pop_percentile[idx] for idx in group])

        # now calculate the amount of energy use in each bin, by subtracting the cumulative energy use 
        # at the lower bound of the bin from the cumulative energy use at the upper bound of the bin,
        # assuming an implict zero for the lower bound of the first bin
        cum_energy_array = np.array([cum_energy_in_country_to_pop_percentile[idx] for idx in group]) 
        # now calculate the amount of energy use in each bin, by subtracting the cumulative energy use 
        # at the lower bound of the bin from the cumulative energy use at the upper bound of the bin,
        # assuming an implict zero for the lower bound of the first bin
        # Assuming your 2D array is called 'array_2d'
        energy_array = np.diff(cum_energy_array)
        zeros_column = np.zeros((energy_array.shape[0], 1))  # Create a column of zeros with same number of rows
        energy_array = np.column_stack((zeros_column, energy_array))    

        # now flatten the pop_binand pc_energy_arrays and sort them both
        #  by the flattened pc_energy_array
        pop_bin = pop_bin.flatten()/pop_group_scalar
        pc_energy_vector = pc_energy_array.flatten()
        energy_vector = energy_array.flatten()

        sorted_idx = np.argsort(pc_energy_vector)
        pop_bin = pop_bin[sorted_idx]
        pc_energy_vector = pc_energy_vector[sorted_idx]
        energy_vector = energy_vector[sorted_idx]
        cum_energy_by_group = np.cumsum(energy_vector)
        cum_pop_by_group = np.cumsum(pop_bin)

        # now create a table with the population and per capita energy use by percentile by creating an interpolation function
        # and then interpolating to the percentile list

        pc_energy_pct_interp = np.interp(percentile_vector, pop_bin, pc_energy_vector)
        cum_energy_pct_interp = np.interp(percentile_vector, pop_bin, cum_energy_by_group)    

        cum_pop_percap_energy_interp = np.interp(energy_level_vector, pc_energy_vector, cum_pop_by_group)
        cum_energy_percap_energy_interp = np.interp(energy_level_vector, pc_energy_vector, cum_energy_by_group)    

        group_percap_energy_pct_interp.append(pc_energy_pct_interp)
        group_cum_energy_pct_interp.append(cum_energy_pct_interp)

        group_cum_pop_percap_energy_interp.append(cum_pop_percap_energy_interp)
        group_cum_energy_percap_energy_interp.append(cum_energy_percap_energy_interp)



    # Step 4: Prepare output data
    out_group_percap_energy_pct = [["per capita energy use by percentile - kW"] + percentile_vector.tolist()] + \
                   [[group_names[idx]] + (group_percap_energy_pct_interp[idx]).tolist() for idx in range(len(groups))]

    out_group_cum_energy_pct = [["cumulative energy use by percentile - kWh"] + percentile_vector.tolist()] + \
                   [[group_names[idx]] + (group_cum_energy_pct_interp[idx]).tolist() for idx in range(len(groups))]

    out_group_cum_pop_level = [["cumulative population by per capita energy use - kW"] + energy_level_vector.tolist()] + \
                   [[group_names[idx]] + (group_cum_pop_percap_energy_interp[idx]).tolist() for idx in range(len(groups))]

    out_group_cum_energy_level = [["cumulative energy use by per capita energy use - kWh"] + energy_level_vector.tolist()] + \
                   [[group_names[idx]] + (group_cum_energy_percap_energy_interp[idx]).tolist() for idx in range(len(groups))]

    # Step 5: Export to Excel with column labels and transposed data
    out_percap_energy_file = f"./{filename_prefix}/{filename_prefix}_group_popPct_pop_{group_type}.xlsx"
    out_cum_energy_file = f"./{filename_prefix}/{filename_prefix}_group_popPct_cumEnergy_{group_type}.xlsx"
    out_cum_pop_level_file = f"./{filename_prefix}/{filename_prefix}_group_popPct_pop_{group_type}.xlsx"
    out_cum_energy_level_file = f"./{filename_prefix}/{filename_prefix}_group_popPct_cumEnergy_{group_type}.xlsx"

    
    # ++++ If this next table is to make sense then the "Population Percentile" should be the population global percentile, sorted by income
    # ++++ I don't think this is the case right now.
    # For per capita energy data by percentile of population
    pc_df = pd.DataFrame(out_group_percap_energy_pct)

    pc_df_transposed = pc_df.transpose()
    # Reset the index to make the first row a column
    pc_df_transposed.reset_index(inplace=True)
    pc_df_transposed.columns = ['GlobalPopulation Percentile'] + list(pc_df_transposed.columns[1:])
    pc_df_transposed.to_excel(out_percap_energy_file, index=False)
    if verbose_level > 0:
        print(f"Exported {out_percap_energy_file}")

    # ++++ This next table is for making a Lorenz curve of cumulative energy use by population percentile
    # For cumulative energy data by percentile of population
    ce_df = pd.DataFrame(out_group_cum_energy_pct)

    ce_df_transposed = ce_df.transpose()
    # Reset the index to make the first row a column
    ce_df_transposed.reset_index(inplace=True)
    ce_df_transposed.columns = ['Global Population Percentile'] + list(ce_df_transposed.columns[1:])
    ce_df_transposed.to_excel(out_cum_energy_file, index=False)
    if verbose_level > 0:
        print(f"Exported {out_cum_energy_file}")

    
    # ++++ This next table is for making a figure showing the cumulative population by per capita energy use
    # For number data by per capita energy use
    pc_df = pd.DataFrame(out_group_cum_pop_level)

    pc_df_transposed = pc_df.transpose()
    # Reset the index to make the first row a column
    pc_df_transposed.reset_index(inplace=True)
    pc_df_transposed.columns = ['Energy Level'] + list(pc_df_transposed.columns[1:])
    pc_df_transposed.to_excel(out_cum_pop_level_file, index=False)
    if verbose_level > 0:
        print(f"Exported {out_cum_pop_level_file}")

    # ++++ This next table is for making a figure showing the cumulative energy use by per capita energy use
    # For cumulative energy data by per capita energy use
    ce_df = pd.DataFrame(out_group_cum_energy_level)

    ce_df_transposed = ce_df.transpose()
    # Reset the index to make the first row a column
    ce_df_transposed.reset_index(inplace=True)
    ce_df_transposed.columns = ['Energy Level'] + list(ce_df_transposed.columns[1:])
    ce_df_transposed.to_excel(out_cum_energy_level_file, index=False)
    if verbose_level > 0:
        print(f"Exported {out_cum_energy_level_file}")

#-------------------------------------------------------------------------------------------------------------


def export_countries_percentile(excel_input_data, percentile_vector, 
                         per_capita_energy_in_country_to_pop_percentile, 
                         cum_energy_in_country_to_pop_percentile, filename_prefix, verbose_level):
    """
    Exports per capita and cumulative energy use by percentile for each country to Excel files.

    Parameters:
        excel_input_data: pandas.DataFrame
            Data containing country names, codes, and energy information.
        percentile_vector: list
            List of percentiles.
        per_capita_energy_in_country_to_pop_percentile: list of lists
            Per capita energy use by population percentile for each country.
        cum_energy_in_country_to_pop_percentile: list of lists
            Cumulative energy use by population percentile for each country.
        filename_prefix: str
            Name used in the output file names.
        verbose_level: int
            Level of verbosity for printing progress and results.

    Returns:
        None
    """
    # Step 1: Prepare per capita energy use data
    country_info = excel_input_data.iloc[:, [1, 2]].values  # Extract country names and codes
    out_pc = [["per capita energy use by percentile", "kW"] + percentile_vector.tolist()] + \
             [list(country_info[idx]) + per_capita_energy_in_country_to_pop_percentile[idx].tolist()
              for idx in range(len(per_capita_energy_in_country_to_pop_percentile))]

    # Step 2: Prepare cumulative energy use data
    out_ce = [["cumulative energy use by percentile", "kW"] + percentile_vector.tolist()] + \
             [list(country_info[idx]) + cum_energy_in_country_to_pop_percentile[idx].tolist()
              for idx in range(len(cum_energy_in_country_to_pop_percentile))]

    # Step 3: Export to Excel with transposed data
    out_percap_energy_file = f"./{filename_prefix}/{filename_prefix}_country_popPct_percapEnergy.xlsx"
    out_cum_energy_file = f"./{filename_prefix}/{filename_prefix}_country_popPct_cumEnergy.xlsx"
    
    # For per capita energy data
    pc_df = pd.DataFrame(out_pc)
    # Use the first row as column headers
    pc_df.columns = pc_df.iloc[0]
    # Remove the first row since it's now the header
    pc_df = pc_df.iloc[1:]
    # Drop the first column
    pc_df = pc_df.iloc[:, 1:]
    # Transpose the DataFrame
    pc_df_transposed = pc_df.transpose()
    # Reset the index to make the first row a column
    pc_df_transposed.reset_index(inplace=True)
    # Use the first row as column headers
    pc_df_transposed.columns = pc_df_transposed.iloc[0]
    # Remove the first row since it's now the header
    pc_df_transposed = pc_df_transposed.iloc[1:]
    pc_df_transposed.to_excel(out_percap_energy_file, index=False)
    if verbose_level > 0:
        print(f"Exported {out_percap_energy_file}")

    # For cumulative energy data
    ce_df = pd.DataFrame(out_ce)
    # Use the first row as column headers
    ce_df.columns = ce_df.iloc[0]
    # Remove the first row since it's now the header
    ce_df = ce_df.iloc[1:]
    # Drop the first column
    ce_df = ce_df.iloc[:, 1:]
    # Transpose the DataFrame
    ce_df_transposed = ce_df.transpose()
    # Reset the index to make the first row a column
    ce_df_transposed.reset_index(inplace=True)
    # Use the first row as column headers
    ce_df_transposed.columns = ce_df_transposed.iloc[0]
    # Remove the first row since it's now the header
    ce_df_transposed = ce_df_transposed.iloc[1:]
    ce_df_transposed.to_excel(out_cum_energy_file, index=False)
    if verbose_level > 0:
        print(f"Exported {out_cum_energy_file}")

#-------------------------------------------------------------------------------------------------------------

def export_country_summary_data(country_summary_data, filename_prefix, verbose_level):
    """
    Exports the country list with relevant headings to an Excel file.

    Parameters:
        country_summary_data: dictionary of lists by country index.
            keys should be: 
            []'Country Name', 'Country Code', 'RMS', 'P', 'Q', 
            'Population', 'GDP', 'Energy', 'Gamma', 'Integral', 'Income Gini', 
            'Energy Gini']

         filename_prefix: str
            Name used in the output file name.
        verbose_level: int
            Level of verbosity for printing progress and results.

    Returns:
        None
    """


    # Create the pandas DataFrame
    df = pd.DataFrame(country_summary_data) # keys are column headings by default

    # Step 3: Export to Excel
    file_name = f"./{filename_prefix}/{filename_prefix}_country_data_various.xlsx"
    df.to_excel(file_name,index=False)

    # Step 4: Print confirmation if verbose
    if verbose_level > 0:
        print(f"Exported {file_name}")

#%%
# main run

def combine_energy_data(country_summary_data, cum_energy_in_country_to_pop_percentile, per_capita_energy_in_country_to_pop_percentile, n_bins_out0):
    """
    Combines energy data and produces aggregated results based on population and energy distribution.

    Parameters:
        country_summary_data: dict
            Dictionary containing country-level data, including population information.
        cum_energy_in_country_to_pop_percentile: numpy.array
            Cumulative energy use by population percentile for each country.
        per_capita_energy_in_country_to_pop_percentile: numpy.ndarray
            Per capita energy use by population percentile for each country.
        n_bins_out0: int 
            Number of bins for output. If 0,  uses the input bin count.

    Returns:
        dict:
            Aggregated energy data as a dictionary with keys corresponding to different metrics.
    """
    # Step 1: Determine the number of bins
    n_bins_in = cum_energy_in_country_to_pop_percentile.shape[1] - 1
    n_bins_out = n_bins_out0 if n_bins_out0 > 0 else n_bins_in

    # Step 2: Calculate increments (energy boundaries for each bin)
    energy_bdry_country = cum_energy_in_country_to_pop_percentile[:, 1:] - cum_energy_in_country_to_pop_percentile[:, :-1]

    # Step 3: Calculate population in each increment
    population_table = (np.expand_dims(country_summary_data["Population"], axis=1) / n_bins_in).repeat(n_bins_in, axis=1)

    # Step 4: Flatten and sort the data
    sorted_table = np.hstack([
        per_capita_energy_in_country_to_pop_percentile[:, 1:].flatten()[:, np.newaxis],
        population_table.flatten()[:, np.newaxis],
        energy_bdry_country.flatten()[:, np.newaxis] # this is the amount of energy to the next lower bin
        ])
    sorted_table = sorted_table[np.argsort(sorted_table[:, 0])]  # Sort by per capita energy

    # Step 5: Compute totals and cumulative fractions
    total_population = np.sum(sorted_table[:, 1])
    total_energy = np.sum(sorted_table[:, 2])
    pop_energy_sort = np.cumsum(sorted_table[:, 1]) / total_population
    energy_energy_sort = np.cumsum(sorted_table[:, 2]) / total_energy
    per_capita_energy_energy_sort = sorted_table[:, 0]

    # Step 6: Interpolate cumulative fractions
    energy_fn = interp1d(pop_energy_sort, np.log(energy_energy_sort), kind="linear", fill_value="extrapolate")
    per_capita_fn = interp1d(pop_energy_sort, np.log(per_capita_energy_energy_sort), kind="linear", fill_value="extrapolate")

    # Step 7: Create output dictionary
    d_idx = 1.0 / n_bins_out
    i_max = np.max(pop_energy_sort)
    i_min = np.min(pop_energy_sort)

    result = {
        "lower_bound": [],
        "higher_bound": [],
        "cumulative_energy_to_high_bound": [],
        "fraction_of_energy_in_bin": [],
        "mean_per_capita_energy_in_bin": [],
        "lower_bound_per_capita_energy": [],
        "higher_bound_per_capita_energy": []
    }

    for idx in range(n_bins_out):
        i_low = max(i_min, idx * d_idx)
        i_high = min(i_max, (idx + 1) * d_idx)
        
        frac_energy_high = np.exp(energy_fn(i_high))
        frac_energy_low = np.exp(energy_fn(i_low))
        cum_energy_in_bin = frac_energy_high - frac_energy_low
        per_capita_energy_in_bin = cum_energy_in_bin * total_energy / (total_population / n_bins_out)
        per_capita_low = np.exp(per_capita_fn(i_low))
        per_capita_high = np.exp(per_capita_fn(i_high))

        result["lower_bound"].append(i_low)
        result["higher_bound"].append(i_high)
        result["cumulative_energy_to_high_bound"].append(frac_energy_high)
        result["fraction_of_energy_in_bin"].append(cum_energy_in_bin)
        result["mean_per_capita_energy_in_bin"].append(per_capita_energy_in_bin)
        result["lower_bound_per_capita_energy"].append(per_capita_low)
        result["higher_bound_per_capita_energy"].append(per_capita_high)

    return result


#-------------------------------------------------------------------------------------------------------------

def export_combined_energy_data(combined_data, filename_prefix):
    """
    Exports combined energy data with appropriate headings to an excel file.

    Parameters:
        combined_data: numpy.ndarray
            Data to be exported, where each sublist/row corresponds to a bin.
        filename_prefix: str
            Name used in the output file name.
        run_name: str
            run_name used in the output file name.

    Returns:
        None
    """
    # Define column headings
    headings = [
        "lower bound",
        "higher bound",
        "cumulative energy to high bound",
        "fraction of energy in bin",
        "mean per capita energy use in bin kW",
        "lower bound per capita energy use kW",
        "upper bound per capita energy use in kW"
    ]

    # Combine headings with data
    #export_data = [headings] + combined_data if isinstance(combined_data, np.ndarray) else [headings] + combined_data


    # Create the output file name
    file_name = f"./{filename_prefix}/{filename_prefix}_global_data_various_b{len(combined_data[list(combined_data.keys())[0]])}.xlsx"

    # Create a pandas DataFrame
    df = pd.DataFrame(combined_data)
    df.columns = headings

    # Export to excel
    df.to_excel(file_name, index=False)

    print(f"Exported {file_name}")

#-------------------------------------------------------------------------------------------------------------

def redistribute(combined_energy_data):
    """
    Computes redistribution scenarios based on combined energy data.

    Parameters:
        combined_energy_data: dict
            Dictionary containing combined energy data with keys:
            - "cumulative_energy_to_high_bound"
            - "fraction_of_energy_in_bin"
            - "mean_per_capita_energy_in_bin"
            - "lower_bound_per_capita_energy"
            - "higher_bound_per_capita_energy"

    Returns:
        numpy.ndarray:
            Redistribution table with columns:
            - Fraction range (lower bound)
            - Fraction range (upper bound)
            - Added energy
            - Freed energy (reversed)
    """
    # Extract values from the dictionary
    cum_energy = np.array(combined_energy_data["cumulative_energy_to_high_bound"])
    fract_energy = np.array(combined_energy_data["fraction_of_energy_in_bin"])
    per_cap_mean = np.array(combined_energy_data["mean_per_capita_energy_in_bin"])
    per_cap_lower = np.array(combined_energy_data["lower_bound_per_capita_energy"])
    per_cap_upper = np.array(combined_energy_data["higher_bound_per_capita_energy"])

    # Calculate mean of per capita energy means
    mean_mean_per_cap = np.mean(per_cap_mean)

    # Number of bins
    n_bins = len(per_cap_mean)

    # Calculate added energy
    added_energy = [
        (per_cap_upper[idx] - np.mean(per_cap_mean[:idx + 1])) * (idx + 1) / n_bins
        for idx in range(n_bins)
    ]
    added_energy = np.array(added_energy) / mean_mean_per_cap

    # Calculate freed energy
    freed_energy_0 = [
        (np.mean(per_cap_mean[n_bins - idx - 1:]) - per_cap_lower[n_bins - idx - 1]) * (idx + 1) / n_bins
        for idx in range(n_bins)
    ]
    freed_energy_0 = np.array(freed_energy_0) / mean_mean_per_cap

    # Adjust freed energy to include normalization
    freed_energy = freed_energy_0 + 1.0 - freed_energy_0[-1]

    # Generate output table
    lower_bound = np.linspace(0, 1 - 1 / n_bins, n_bins)
    upper_bound = np.linspace(1 / n_bins, 1, n_bins)

    return np.column_stack((lower_bound, upper_bound, added_energy, freed_energy[::-1]))

#-------------------------------------------------------------------------------------------------------------

def export_redist_energy_data(redist, filename_prefix):
    """
    Exports redistribution energy data with appropriate headings to an excel file.

    This does two calculations:
    1. How much energy would you need to add globally to bring the lowest X% of energy users up to the energy
         use rate of the Xth percentile?
    2. How much energy would you save globally if the highest X% of energy users reduced their energy use to the
    Parameters:
        redist: numpy.ndarray
            Redistribution data as a table with columns:
            - Lower bound
            - Higher bound
            - Fraction of global energy supply required
            - Fraction of global energy supply made available
        filename_prefix: str
            Name used in the output file name.
        run_name: str
            run_name used in the output file name.

    Returns:
        None
    """
    # Define column headings
    headings = [
        "lower bound",
        "higher bound",
        "fraction of global energy supply required to bring everyone "
        "below this level to the upper bound of this level",
        "fraction of global energy supply made available if everyone "
        "above this level came down to the per capita energy use of this lower bound"
    ]

    # Create the output file name
    file_name = f"./{filename_prefix}/{filename_prefix}_global_redist_various_b{len(redist)}.xlsx"

    df = pd.DataFrame(redist)
    df.columns = headings

    # Export to excel
    df.to_excel(file_name, index=False)

    print(f"Exported {file_name}")


#%%


# run code

if __name__ == "__main__":
    # Code to run only when executed from the terminal

    #--------------------------------------------------------------------------
    # Key run time parameters
    run_name = "test"  # Name of the run
    n_percentile_levels = 10000  # 100 for testing, 10000 for production
    n_energy_levels = 1000 # 100 for test and summary output, 1000 for production
    n_bins_out = 1000 # number of bins for output
    verbose_level = 2
    dir = r"C:\Users\kcaldeira\github\energy_distribution"
    #data_input_file_name = "Energy Distribution Input (2022) 2025-02-05.xlsx"
    #data_input_file_name = "Energy Distribution Input 2025-03-04 - Test.xlsx"
    #data_input_file_name = "Energy Distribution Input 2025-03-04 - Test2.xlsx"
    #data_input_file_name = "Energy Distribution Input 2025-03-04 - Test3.xlsx" 
    #data_input_file_name = "Energy Distribution Input 2025-03-04 - Test4.xlsx" # only 5 countries, gamma = 0.5
    #data_input_file_name = "Energy Distribution Input 2025-03-04 - Test6.xlsx" # only 10 countries, 2x5, gamma = 0.5
    #data_input_file_name = "Energy Distribution Input (2021) 2025-02-05.xlsx"
    #data_input_file_name = "Energy Distribution Input (2020) 2025-02-05.xlsx"
    #data_input_file_name = "Energy Distribution Input (2019) 2025-02-05.xlsx"
    #data_input_file_name = "Energy Distribution Input (2018) 2025-02-05.xlsx"
    #data_input_file_name = "Data-input_Dioha-et-al_2022-08-05.xlsx"
    #data_input_file_name = "Energy Distribution Input (for Ken) 2025-01-24.xlsx"
    data_input_file_name = "Energy Distribution Input (for Ken) 2025-01-24 18-test.xlsx"
    epsilon = 1e-12  # Approximately one-hundredth of a person for 10^10 people
    date_stamp = datetime.now().replace(second=0, microsecond=0).isoformat().replace(':', '-').replace('T', '_')[:-3]
   #--------------------------------------------------------------------------

       # Import the Excel file
    excel_input_data = pd.read_excel(data_input_file_name, sheet_name=0)

    # Get the dimensions of the data frame
    dimensions = excel_input_data.shape
    print(f"Dimensions of the input data: {dimensions}")

    # Step 0: Compute gamma (global elasticity of energy use)
    gamma_res = compute_elasticity_of_energy_use(excel_input_data)
    gamma = gamma_res[0]
    if verbose_level > 0: print(f"Gamma: {gamma}")

    # within country gamma = between country gamma
    run_energy_dist(excel_input_data, gamma, n_percentile_levels, n_energy_levels, n_bins_out, verbose_level,  epsilon, run_name, dir, date_stamp)
    #gamma = 0.0
    #run_energy_dist(excel_input_data, 0.0, n_percentile_levels, n_energy_levels, n_bins_out, verbose_level,  epsilon, run_name, dir, date_stamp)
    #gamma = 0.5
    #run_energy_dist(excel_input_data, 0.5, n_percentile_levels, n_energy_levels, n_bins_out, verbose_level,  epsilon, run_name, dir, date_stamp)
    #gamma = 1.0
    #run_energy_dist(excel_input_data, 1.0, n_percentile_levels, n_energy_levels, n_bins_out, verbose_level,  epsilon, run_name, dir, date_stamp)

# usage:
# python.exe -i "c:/Users/kcaldeira/My Drive/Edgar distribution/energy_dist.py"
# %%
