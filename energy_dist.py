# translate version of Mathematica code

#%% Import packages

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import math
import statsmodels.api as sm
import time
import platform, os
from scipy.special import beta as beta_func, betainc

#%% main processing code

def create_key_variables(country_list, percentile_list, n_energy_levels,  verbose_level):
    """
    Processes input data to compute various energy-related tables.
    
    Parameters:
        input_data: str
            Path to the Excel file containing country-level data.
        percentile_list: list
            List of percentiles (e.g., np.arange(0, 1.01, 0.01)).
        filename_prefix: str
            Text string to incorporate in output file names.
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
    per_capita_energy_bdry_country, cum_energy_bdry_country = gen_country_lists_by_fract_of_pop(
        country_list, percentile_list,epsilon, verbose_level
    )

    # Step 3: Create master table of evenly spaced energy levels
    if verbose_level > 0:
        print("Generating list by country and energy use level")
    energy_level_list = np.logspace(-2, 3, num=n_energy_levels).tolist()

    # Step 4: Compute fraction of population for energy levels
    if verbose_level > 0:
        print(f"Computing fractPopTable; {datetime.now()}")
    fract_pop_table = fract_pop_in_country_to_energy_per_capita_level(
        per_capita_energy_bdry_country, percentile_list, energy_level_list
    )

    # Step 5: Compute population by energy level
    pop_table = np.array(country_list["Population"])[:, np.newaxis] * fract_pop_table

    # Step 6: Compute cumulative energy use by energy level
    if verbose_level > 0:
        print(f"Computing energyTable; {datetime.now()}")
    energy_table = energy_in_country_to_fract_pop_level(
        cum_energy_bdry_country, percentile_list, fract_pop_table
    )

    # Return the results in a dictionary
    return {
        "country_list":country_list,
        "per_capita_energy_bdry_country": per_capita_energy_bdry_country,
        "cum_energy_bdry_country": cum_energy_bdry_country,
        "energy_level_list": energy_level_list,
        "fract_pop_table": fract_pop_table,
        "pop_table": pop_table,
        "energy_table": energy_table
    }

#----------------------------------------------------------------------------------------------

def prep_country_level_data(input_data, gamma, epsilon, verbose_level):
    """
    Prepares country-level data by computing parameters like gamma, Lorenz curve fits,
    and Gini coefficients for income and energy distribution.

    Parameters:
        input_data: pandas.DataFrame
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

    # Step 2: Compute country-level parameters for fits to income Lorenz curve
    lorenz_interpolation_list_country = [find_lorenz_interpolation_country(row, verbose_level) for row in input_data.iloc[1:].to_numpy()]

    # Step 3: Get country-level population, GDP, and energy data
    pop_list = 1000.0 * input_data.iloc[1:, 4].to_numpy()
    gdp_list = input_data.iloc[1:, 5].to_numpy()
    energy_list = (10**9 / (8760 * 3600)) * input_data.iloc[1:, 6].to_numpy()

    # Step 4: Compute integral list for energy Lorenz bin
    integral_list = [
        energy_in_lorenz_range(0., 1.-epsilon, rmspq[1], rmspq[2], gamma, 1, 1, epsilon)
        for rmspq in lorenz_interpolation_list_country
    ]

    # Step 5: Compute income Gini list
    income_gini_list = [
        1 - integral_income_lorenz(rmspq[1], rmspq[2]) / 0.5
        for rmspq in lorenz_interpolation_list_country
    ]

   # Step 6: Compute energy Gini list
    energy_gini_list = [
        1 - integral_energy_lorenz(lorenz_interpolation_list_country[idx][1], lorenz_interpolation_list_country[idx][2], gamma, 1, integral_list[idx],epsilon) / 0.5
        for idx  in range(len(lorenz_interpolation_list_country))
    ]


    # Step 7: Create country summary table
    country_summary_data = {
        "Country Name": input_data.iloc[1:, 1].to_numpy().tolist(),
        "Country Code": input_data.iloc[1:, 2].to_numpy().tolist(),
        "RMS": [rmspq[0] for rmspq in lorenz_interpolation_list_country],
        "P": [rmspq[1] for rmspq in lorenz_interpolation_list_country],  # Extract P (pp) into its own list
        "Q": [rmspq[2] for rmspq in lorenz_interpolation_list_country],  # Extract Q (qq) into its own list
        "Population": pop_list.tolist(),
        "GDP": gdp_list.tolist(),
        "Energy": energy_list.tolist(),
        "Gamma": [gamma] * len(pop_list),
        "Integral": integral_list,
        "Income Gini": income_gini_list,
        "Energy Gini": energy_gini_list,
    }


    return country_summary_data

#-----------------------------------------

def find_lorenz_interpolation_country(input_data, epsilon, verbose):
    """
    Fits parameters pp and qq for a country using income Lorenz curve data.

    Parameters:
        input_data_country: list or numpy array
            One row of input data corresponding to a country.
        epsilon: float
            Small value used for tolerances, etc
        verbose: integer
            Whether to print detailed information.

    Returns:
        tuple:
            (rms, p, q): Root mean square error, and fitted parameters p (pp) and q (qq).
    """
   
    # Population and cumulative income levels
    cum_pop_levels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    cum_inc_levels = np.cumsum(input_data.iloc[:,8,18].to_numpy())

    # Note the assumption is that the last column of cum_inc_levels is equal to 1

    curve_fit = np.array([find_lorenz_interpolation(cum_pop_levels, row) for row in cum_inc_levels])

    # Objective function to minimize
    def objective(params):
        pp, qq = params
        y_test = np.zeros_like(y)
        y_test[1:] = x[1:]**pp * (1 - (1 - x[1:])**qq)
        # Compute RMS error
        error = pop_levels * ((np.diff(y_test) / np.diff(y)) - 1)**2
        return np.sum(error)

    # Initial guesses and bounds for pp and qq
    initial_guess = [0.8, 0.6]
    bounds = [(epsilon, 1 - epsilon), (epsilon, 1 - epsilon)]

    # Minimize the objective function
    result = minimize(objective, initial_guess, bounds=bounds, tol =1e-14)

    # Extract the results
    rms = result.fun
    p = abs(result.x[0])
    q = abs(result.x[1])

    # Verbose output
    if verbose:
        print(f"RMS: {rms}, pp: {p}, qq: {q}")

    return rms, p, q

#----------------------------------------------------------------------------------------------------
#-----------CODE TO DO COUNTRY LEVEL FITS -----------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# find jantzen volpert fit going through two points


def jantzen_volpert_fn(x, p, q):
    return x**p * (1 - (1 - x)**q)

def jantzen_volpert_fn_deriv(x, p, q):
    term1 = p * x**(p - 1) * (1 - (1 - x)**q)
    term2 = x**p * q * (1 - x)**(q - 1)
    return term1 + term2

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

def fit_monotonic_convex_spline_with_derivatives(x_data, y_data, dy_dx_start, dy_dx_end):
    """
    Fits a cubic spline ensuring:
    - Monotonic increasing function
    - First derivative is continuous and increasing
    - Uses given first derivatives at the endpoints
    """
    # Fit a cubic spline with first derivative constraints at the endpoints
    spline = CubicSpline(x_data, y_data, bc_type=((1, dy_dx_start), (1, dy_dx_end)), extrapolate=False)
    
    return spline

def find_lorenz_interpolation(x_data, y_data):
    """
    Process x_data and y_data to fit Jantzen-Volpert function to the first and last pair of points,
    compute derivatives, and fit a cubic spline through the intermediate points.
    """
    # Fit Jantzen-Volpert function to first and last pair of points
    p_start, q_start = find_jantzen_volpert_p_q(x_data[0], y_data[0], x_data[1], y_data[1])
    p_end, q_end = find_jantzen_volpert_p_q(x_data[-2], y_data[-2], x_data[-1], y_data[-1])
    
    # Compute derivatives at the second and next-to-last data points
    dy_dx_start = jantzen_volpert_fn_deriv(x_data[1], p_start, q_start)
    dy_dx_end = jantzen_volpert_fn_deriv(x_data[-2], p_end, q_end)
    
    # Fit the spline using second to next-to-last points
    spline = fit_monotonic_convex_spline_with_derivatives(x_data[1:-1], y_data[1:-1], dy_dx_start, dy_dx_end)
    
    return spline, p_start, q_start, p_end, q_end

def compute_income_lorenz_integral(x_data, spline, p_left, q_left, p_right, q_right):
    
    # Integrate the left analytic segment from 0 to x_data[1].
    integral_left = jantzen_volpert_fn_integral(0, x_data[1],p_left, q_left)
    
    # Integrate the middle spline segment using its antiderivative.
    spline_antideriv = spline.antiderivative()
    integral_middle = spline_antideriv(x_data[-2]) - spline_antideriv(x_data[1])
    
    # Integrate the right analytic segment from x_data[-2] to 1.
    integral_right =  jantzen_volpert_fn_integral(x_data[-2], 1, p_right, q_right)
    
    # Sum the three pieces to get the total integral.
    total_integral = integral_left + integral_middle + integral_right

    return total_integral

def compute_energy_lorenz_integral(x_data, spline, p_left, q_left, p_right, q_right, gamma):
    """
    Integrates f(x)^gamma over [0, 1] where f(x) is defined piecewise:
      - For x in [0, x_data[1]]: f(x) = jantzen_volpert_fn(x, p_left, q_left)
      - For x in [x_data[1], x_data[-2]]: f(x) = spline(x)
      - For x in [x_data[-2], 1]: f(x) = jantzen_volpert_fn(x, p_right, q_right)
      
    Parameters:
        x_data : array-like
            Array of x-values spanning [0, 1] (with x_data[0] == 0 and x_data[-1] == 1).
        spline : CubicSpline
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
    integral_left, _ = quad(lambda x: jantzen_volpert_fn(x, p_left, q_left)**gamma,
                            0, x_data[1])
    
    # Integrate the middle spline segment numerically from x_data[1] to x_data[-2]:
    integral_middle, _ = quad(lambda x: spline(x)**gamma, x_data[1], x_data[-2])
    
    # Integrate the right analytic segment from x_data[-2] to 1:
    integral_right, _ = quad(lambda x: jantzen_volpert_fn(x, p_right, q_right)**gamma,
                             x_data[-2], 1)
    
    # Sum the three pieces to get the total integral:
    total_integral = integral_left + integral_middle + integral_right
    return total_integral

def compute_d_income_lorenz_dx(x, x_data, spline, p_left, q_left, p_right, q_right):
    """
    Computes the derivative of the piecewise function f(x) at a given x, where f(x)
    is defined as:
      - For x in [0, x_data[1]]:
            f(x) = jantzen_volpert_fn(x, p_left, q_left)
      - For x in [x_data[1], x_data[-2]]:
            f(x) = spline(x)
      - For x in [x_data[-2], 1]:
            f(x) = jantzen_volpert_fn(x, p_right, q_right)
            
    Parameters:
        x : float
            The point at which the derivative is evaluated.
        x_data : array-like
            Array of x-values spanning [0, 1] (with x_data[0] == 0 and x_data[-1] == 1).
        spline : CubicSpline
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
    elif x < x_data[-2]:
        # Compute the derivative using the spline's derivative.
        return spline.derivative()(x)
    
    # Right analytic segment: x_data[-2] <= x <= 1
    else:
        return jantzen_volpert_fn_deriv(x, p_right, q_right)
    
def compute_d_energy_lorenz_dx(x, x_data, spline, p_left, q_left, p_right, q_right, gamma):
    """
    Computes the derivative of f(x)^gamma with respect to x for 0 <= x <= 1, where f(x) is defined piecewise as:
      - For x in [0, x_data[1]]:
            f(x) = jantzen_volpert_fn(x, p_left, q_left)
      - For x in [x_data[1], x_data[-2]]:
            f(x) = spline(x)
      - For x in [x_data[-2], 1]:
            f(x) = jantzen_volpert_fn(x, p_right, q_right)
    
    The derivative is computed as:
    
        d/dx (f(x)^gamma) = gamma * f(x)^(gamma-1) * f'(x)
    
    Parameters:
        x : float
            The point at which to evaluate the derivative.
        x_data : array-like
            Array of x-values spanning [0, 1].
        spline : CubicSpline
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
        f_val = jantzen_volpert_fn(x, p_left, q_left)
        f_prime = jantzen_volpert_fn_deriv(x, p_left, q_left)
    elif x < x_data[-2]:
        f_val = spline(x)
        f_prime = spline.derivative()(x)
    else:
        f_val = jantzen_volpert_fn(x, p_right, q_right)
        f_prime = jantzen_volpert_fn_deriv(x, p_right, q_right)
    
    return gamma * f_val**(gamma - 1) * f_prime

#----------------------------------------------------------------------------------------------------
#-----------CODE ABOVE NOTE USED --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

def integral_energy_lorenz(pp, qq, gamma, energy, energy_integral,epsilon):
    """
    Computes the integral of the energy Lorenz bin over the range [0, 1].

    Parameters:
        pp: float
            Parameter pp for the Lorenz curve.
        qq: float
            Parameter qq for the Lorenz curve.
        gamma: float
            Elasticity parameter.
        energy: float
            Energy level.
        energy_integral: float
            Pre-computed integral of energy.

    Returns:
        float
            The result of the integral over the specified range.
    """
    # Function to integrate
    def integrand(x0):
        return energy_in_lorenz_range(0, x0, pp, qq, gamma, energy, energy_integral,epsilon)

    # Perform the numerical integration
    result, _ = quad(integrand, 0, 1, epsrel = epsilon)  # Integrate over the range [0, 1]
    return result

#-------------------------------------------------------------------------------------------------------------

def energy_per_capita_fn(x, pp, qq, gamma, pop, energy, energy_integral):
    """
    Computes the energy per capita function based on the given parameters.

    Parameters:
        x: float
            A scalar value, must be numeric.
        pp: float
            Parameter pp for the Lorenz curve.
        qq: float
            Parameter qq for the Lorenz curve.
        gamma: float
            Elasticity parameter.
        pop: float
            Population size.
        energy: float
            Energy level.
        energy_integral: float
            Pre-computed integral of energy.

    Returns:
        float
            The energy per capita value.
    """

    # Compute the main term
    numerator = (
        energy / pop
    ) * (
        (pp * (1 - (1 - x)**qq) * x**(-1 + pp) +
         qq * (1 - x)**(-1 + qq) * x**pp)**gamma
    )

    # Replace indeterminate values with 0 and complex infinity with infinity
    if numerator == float('inf') or numerator == float('nan') or energy_integral == 0:
        return 0
    return numerator / energy_integral

#-------------------------------------------------------------------------------------------------------------

def energy_in_lorenz_range(x0, x1, pp, qq, gamma, energy, energy_integral, epsilon):
    """
    Computes the energy Lorenz bin integral over a specified range.

    Parameters:
        x0: float
            Lower bound of the integration range.
        x1: float
            Upper bound of the integration range.
        pp: float
            Parameter pp for the Lorenz curve.
        qq: float
            Parameter qq for the Lorenz curve.
        gamma: float
            Elasticity parameter.
        energy: float
            Energy level.
        energy_integral: float
            Pre-computed integral of energy.

    Returns:
        float
            The value of the energy Lorenz bin for the given range.
    """

    # Define the integrand
    def integrand(x):
        return (pp * (1 - (1 - x)**qq) * x**(-1 + pp) +
                qq * (1 - x)**(-1 + qq) * x**pp)**gamma

    # Perform the numerical integration over [x0, x1]
    integral_result, _ = quad(integrand, x0, x1, epsrel = epsilon)

    # Scale the result by energy and energyIntegral
    if energy_integral == 0:
        return 0
    return (energy / energy_integral) * integral_result




#-------------------------------------------------------------------------------------------------------------

def gen_country_lists_by_fract_of_pop(country_list, percentile_list, epsilon, verbose_level):
    """
    Generates lists of per capita energy use and cumulative energy use
    by population percentile for each country.

    Parameters:
        country_list: dict
            The dictionary containing country-level data.
        percentile_list: list
            A list of percentiles (e.g., np.arange(0, 1.01, 0.01)).
        filename_prefix: str
            Name to be used in output (not used directly here but passed).
        verbose_level: int
            Level of verbosity for print statements.

    
            - per_capita_energy_bdry_country: numpy array of per capita energy use.
            - cum_energy_bdry_country: numpy of cumulative energy use.
    """
    if verbose_level > 0:
        print("Generating lists of per capita and cumulative energy use by population percentile...")

    # Generate per capita energy use by population percentile
    per_capita_energy_bdry_country = np.array([
        [
            energy_per_capita_fn(
                x, 
                country_list["P"][idx_country], 
                country_list["Q"][idx_country], 
                country_list["Gamma"][idx_country], 
                country_list["Population"][idx_country], 
                country_list["Energy"][idx_country], 
                country_list["Integral"][idx_country]
            )
            for x in percentile_list
        ]
        for idx_country in range(len(country_list["Population"]))
        ])

    # Generate cumulative energy use by population percentile
    cum_energy_bdry_country = np.array([
        np.cumsum([
            energy_in_lorenz_range(
                percentile_list[max(0, i - 1)],
                percentile_list[i],
                country_list["P"][idx_country],
                country_list["Q"][idx_country],
                country_list["Gamma"][idx_country],
                country_list["Energy"][idx_country],
                country_list["Integral"][idx_country],
                epsilon
            )
            for i in range(len(percentile_list))
        ])
        for idx_country in range(len(country_list["Population"]))
        ])

    if verbose_level > 0:
        print("Completed generating energy use lists.")

    return per_capita_energy_bdry_country, cum_energy_bdry_country

#-------------------------------------------------------------------------------------------------------------

def fract_pop_in_country_to_energy_per_capita_level(per_capita_energy_bdry_country, pct_list, e_val_list):
    """
    Computes the fraction of the population by energy per capita level.

    Parameters:
        per_capita_energy_bdry_country: np.array
            A list where each element corresponds to a country's per capita energy boundaries.
        pct_list: list
            A list of percentiles.
        e_val_list: list
            A list of energy values to map to percentiles.

    Returns:
        np.array
            A list where each element corresponds to a country's fraction of population mapped to energy levels.
    """
    result = []
    for idx in range(len(per_capita_energy_bdry_country)):
        # Extract per capita energy boundaries for the current country
        per_capita_energy = per_capita_energy_bdry_country[idx]
        
        # Determine the minimum and maximum energy levels
        min_energy = min(per_capita_energy)
        max_energy = max(per_capita_energy)
        
        # Create a linear interpolation function
        interp_fn = interp1d(per_capita_energy, pct_list, bounds_error=False, fill_value=(pct_list[0], pct_list[-1]))
        
        # Map energy values to percentiles, clamping them to the [min_energy, max_energy] range
        mapped_values = [interp_fn(np.clip(e_val, min_energy, max_energy)) for e_val in e_val_list]
        
        # Append the mapped values for the current country
        result.append(mapped_values)
    
    return np.array(result)

#-------------------------------------------------------------------------------------------------------------

def energy_in_country_to_fract_pop_level(cum_energy_bdry_country, pct_list, pop_fract_list):
    """
    Computes the energy use mapped to fraction of population levels.

    Parameters:
        cum_energy_bdry_country: np.array
            A list where each element corresponds to a country's cumulative energy boundaries.
        pct_list: list
            A list of percentiles.
        pop_fract_list: list of lists
            A list of lists, where each sublist corresponds to the fraction of population for a specific country.

    Returns:
        list of lists:
            A list where each element corresponds to a country's energy use mapped to population levels.
    """
    energy_table = []
    for idx in range(len(cum_energy_bdry_country)):
        # Extract cumulative energy boundaries for the current country
        cum_energy = cum_energy_bdry_country[idx]
        
        # Determine the minimum and maximum energy levels
        min_energy = np.min(cum_energy)
        max_energy = np.max(cum_energy)
        
        # Create a linear interpolation function
        interp_fn = interp1d(pct_list, cum_energy, bounds_error=False, fill_value=(cum_energy[0], cum_energy[-1]))
        
        # Map population fractions to energy values, clamping them to the [min_energy, max_energy] range
        mapped_values = [interp_fn(np.clip(pop_frac, min_energy, max_energy)) for pop_frac in pop_fract_list[idx]]
        
        # Append the mapped values for the current country
        energy_table.append(mapped_values)
    
    return np.array(energy_table)
#-------------------------------------------------------------------------------------------------------------

def integral_income_lorenz(p, q):
    """
    Computes the integral of the income Lorenz curve.

    Parameters:
        p: float
            Parameter p of the Lorenz curve.
        q: float
            Parameter q of the Lorenz curve.

    Returns:
        float
            The result of the Lorenz curve integral.
    """
    return 1 / (1 + p) - (math.gamma(1 + p) * math.gamma(1 + q)) / math.gamma(2 + p + q)

#-------------------------------------------------------------------------------------------------------------

def compute_elasticity_of_energy_use(input_data):
    """
    Computes the elasticity of energy use based on input data.

    Parameters:
        input_data: pandas.DataFrame
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
    pop = 1000 * input_data.iloc[1:, 4].to_numpy()  # Population in number of people (2019)
    gdp = input_data.iloc[1:, 5].to_numpy()         # GDP in 2017 dollars (2019)
    energy = (10**9 / 8760) * input_data.iloc[1:, 6].to_numpy()  # Energy use in kW (average)

    # Compute per capita values
    log_gdp_per_capita = np.log(gdp / pop)
    log_energy_per_capita = np.log(energy / pop)

    # Prepare the regression model
    X = sm.add_constant(log_gdp_per_capita)  # Add intercept term
    y = log_energy_per_capita
    weights = pop

    # Fit a weighted least squares regression
    model = sm.WLS(y, X, weights=weights).fit()

    # Extract results
    elasticity_coefficient = model.params[1]  # Slope (elasticity)
    adjusted_r_squared = model.rsquared_adj   # Adjusted R-squared
    slope_std_error = model.bse[1]            # Standard error of the slope

    return elasticity_coefficient, adjusted_r_squared, slope_std_error

#-------------------------------------------------------------------------------------------------------------

"""
Note that the underlying Lorenz curve for cumulative fraction of income as a cumulative fraction of population in the Jantzen and Volpert (2013) approach is:

L[x] == x^p ( 1 - (1-x)^q)

If pop is the country population and gdp is total income for the population, the per capita income <incomePerCapita> at population fraction x is

incomePerCapita[x] == (gdp / pop) * d L[x] / d x = p (1-(1-x)^q) x^(-1+p)+q (1-x)^(-1+q) x^p

if energy is country total energy use and gamma is the elasticity of energy use with population, we can define the variable <energyIntegral>

energyIntegral = Integral[ (p (1-(1-x)^q) x^(-1+p)+q (1-x)^(-1+q) x^p)^gamma, {x, 0 1}]

then per capita energy use <energyPerCapita> at population fraction x is:

energyPerCapita[x] == (energy / pop) (p (1-(1-x)^q) x^(-1+p)+q (1-x)^(-1+q) x^p)^gamma  / integral

and cumulative energy to population fraction x is

energyCumulative[x] ==  energy * Integral[ (p (1-(1-x0)^q) x0^(-1+p)+q (1-x0)^(-1+q) x0^p)^gamma, {x0, 0 1}] / integral

"""

#%%
# Code to summarize key variables and output as xlsx files

def find_country_groups_per_capita(input_data, n_groups, idx_group):
    """
    Groups countries by per capita income or energy use.

    Parameters:
        input_data: pandas.DataFrame
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
    input_data["PerCapita"] = input_data.iloc[1:, idx_group] / input_data.iloc[1:, 4]  # Per capita value (e.g., income/pop)
    sorted_data = input_data.iloc[1:].sort_values(by="PerCapita").reset_index(drop=True)

    # Step 2: Calculate cumulative population, income, and energy
    cum_pop = sorted_data.iloc[:, 4].cumsum()
    cum_pop /= cum_pop.iloc[-1]  # Normalize to [0, 1]

    cum_income = sorted_data.iloc[:, 5].cumsum()
    cum_income /= cum_income.iloc[-1]  # Normalize to [0, 1]

    cum_energy = sorted_data.iloc[:, 6].cumsum()
    cum_energy /= cum_energy.iloc[-1]  # Normalize to [0, 1]

    # Step 3: Determine country positions corresponding to population groups
    pop_targets = np.linspace(1 / n_groups, 1, n_groups)
    country_positions = [np.searchsorted(cum_pop, target) for target in pop_targets]

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

def find_country_group_indices(country_groups, country_list):
    """
    Finds the indices of countries in the country list that belong to each group.

    Parameters:
        country_groups: list of lists
            A list where each sublist contains country identifiers (e.g., country codes) for a specific group.
        country_list: dict
            A dictionary containing country-level data, where "Country Code" is a key referencing country identifiers.

    Returns:
        list of lists:
            A list where each sublist contains the indices of countries in the `country_list` that belong to the respective group.
    """
    country_codes = country_list["Country Code"]  # Extract country codes from the country list

    # Find indices for each group
    group_indices = [
        [idx for idx, code in enumerate(country_codes) if code in group]
        for group in country_groups
    ]

    return group_indices


#-------------------------------------------------------------------------------------------------------------

def export_groups_percentile(groups, group_names, group_type, input_data, 
                      percentile_list, per_capita_energy_bdry_country, 
                      cum_energy_bdry_country, filename_prefix, verbose_level):
    """
    Exports per capita and cumulative energy use by percentile for each group to Excel files.

    Parameters:
        groups: list of lists
            Indices of countries in each group.
        group_names: list
            Names of the groups.
        group_type: str
            Type of the group, used in file names.
        input_data: pandas.DataFrame
            Data containing population and energy information for countries.
        percentile_list: list
            List of percentiles.
        per_capita_energy_bdry_country: list of lists
            Per capita energy use by population percentile for each country.
        cum_energy_bdry_country: list of lists
            Cumulative energy use by population percentile for each country.
        filename_prefix: str
            Name used in the output file names.
        verbose_level: int
            Level of verbosity for printing progress and results.

    Returns:
        None
    """
    # Step 1: Compute population groups
    ### CHECK ON INDIEXING
    population_groups = [
        [input_data.iloc[ 1+idx, 4] for idx in group] for group in groups
    ]

    # Step 2: Calculate group per capita energy use
    group_pc = [
        sum(
            np.array(population_groups[idx])[:,np.newaxis] *
            np.array(per_capita_energy_bdry_country)[groups[idx]]
        ) / sum(population_groups[idx])
        for idx in range(len(groups))
    ]

    # this next line had a bug in the original code
    group_ce = [
       [ sum(cum_energy_bdry_country[idx]) for idx in group]
        for group in groups
        ]

    # Step 4: Prepare output data
    out_group_pc = [["per capita energy use by percentile - kW"] + percentile_list] + \
                   [[group_names[idx]] + group_pc[idx].tolist() for idx in range(len(groups))]

    out_group_ce = [["cumulative energy use by percentile - kW"] + percentile_list] + \
                   [[group_names[idx]] + group_ce[idx] for idx in range(len(groups))]

    # Step 5: Export to Excel
    out_pc_file = f"./{filename_prefix}/{filename_prefix}_popPct_pop_{group_type}.xlsx"
    out_ce_file = f"./{filename_prefix}/{filename_prefix}_group_popPct_cumEnergy_{group_type}.xlsx"

    
    pd.DataFrame(out_group_pc).to_excel(out_pc_file, index=False, header=False)
    if verbose_level > 0:
        print(f"Exported {out_pc_file}")

    pd.DataFrame(out_group_ce).to_excel(out_ce_file, index=False, header=False)
    if verbose_level > 0:
        print(f"Exported {out_ce_file}")


#-------------------------------------------------------------------------------------------------------------

def export_groups_pc_energy(groups, group_names, group_type, energy_level_list, pop_table, energy_table, filename_prefix, verbose_level):
    """
    Exports group population and energy data at different energy levels to Excel files.

    Parameters:
        groups: list of lists
            Indices of countries in each group.
        group_names: list
            Names of the groups.
        group_type: str
            Type of the group, used in file names.
        energy_level_list: list
            List of energy levels.
        pop_table: numpy array
            Table of population data by energy levels.
        energy_table: numpy array
            Table of cumulative energy data by energy levels.
        filename_prefix: str
            Name used in the output file names.
        verbose_level: int
            Level of verbosity for printing progress and results.

    Returns:
        None
    """
    print ("shape of pop_table", pop_table.shape)
    print ("shape of energy_table", energy_table.shape)
    print ("shape of groups", len(groups))
    print ("shape of groups[0]", len(groups[0]))
    print (groups[0])
    print ("shape of groups[1]", len(groups[1]))
    print (groups[1])
    print ("shape of groups[2]", len(groups[2]))
    print (groups[2])
    print ("shape of groups[3]", len(groups[3]))
    print (groups[3])
    print ("shape of pop_table[groups[0]]", pop_table[groups[0]].shape)
    print (groups[4])

    print ("shape of group_names", len(group_names))

    # Step 1: Compute total population and energy for each group
    group_pop = [np.sum(pop_table[group],axis=0) for group in groups]
    group_energy = [np.sum(energy_table[group],axis=0) for group in groups]

    # Step 2: Prepare output for population to energy levels
    out_group_pc = [["population to energy level"] + energy_level_list] + \
                   [[group_names[idx]] + list(group_pop[idx]) for idx in range(len(groups))]

    # Prepare output for cumulative energy to energy levels
    out_group_ce = [["cumulative energy to energy level - kW"] + energy_level_list] + \
                   [[group_names[idx]] + list(group_energy[idx]) for idx in range(len(groups))]

    # Step 3: Export the first two tables to Excel
    out_pc_file = f"./{filename_prefix}/{filename_prefix}_group_energyLevel_pop_{group_type}.xlsx"
    out_ce_file = f"./{filename_prefix}/{filename_prefix}_group_energyLevel_cumEnergy_{group_type}.xlsx"
    
    pd.DataFrame(out_group_pc).to_excel(out_pc_file, index=False, header=False)
    if verbose_level > 0:
        print(f"Exported {out_pc_file}")

    pd.DataFrame(out_group_ce).to_excel(out_ce_file, index=False, header=False)
    if verbose_level > 0:
        print(f"Exported {out_ce_file}")

    # Step 4: Calculate geometric mean of energy levels and bin data
    energy_level_list_gm = np.sqrt(np.array(energy_level_list[1:]) * np.array(energy_level_list[:-1]))
    group_pop_bin = np.diff(group_pop, axis=1)
    group_energy_bin = np.diff(group_energy, axis=1)

    # Prepare output for population in energy bins
    out_group_pop_bin = [["population in bin"] + list(energy_level_list_gm)] + \
                        [[group_names[idx]] + list(group_pop_bin[idx]) for idx in range(len(groups))]

    # Prepare output for energy in energy bins
    out_group_energy_bin = [["energy in bin - kW"] + list(energy_level_list_gm)] + \
                           [[group_names[idx]] + list(group_energy_bin[idx]) for idx in range(len(groups))]

    # Step 5: Export the bin data to Excel
    out_pop_bin_file = f"./{filename_prefix}/{filename_prefix}_group_energyLevel_pop.xlsx"
    out_energy_bin_file = f"./{filename_prefix}/{filename_prefix}_group_energyLevel_energy.xlsx"
 
    
    pd.DataFrame(out_group_pop_bin).to_excel(out_pop_bin_file, index=False, header=False)
    if verbose_level > 0:
        print(f"Exported {out_pop_bin_file}")

    pd.DataFrame(out_group_energy_bin).to_excel(out_energy_bin_file, index=False, header=False)
    if verbose_level > 0:
        print(f"Exported {out_energy_bin_file}")

#-------------------------------------------------------------------------------------------------------------

def export_countries_percentile(input_data, percentile_list, 
                         per_capita_energy_bdry_country, 
                         cum_energy_bdry_country, filename_prefix, verbose_level):
    """
    Exports per capita and cumulative energy use by percentile for each country to Excel files.

    Parameters:
        input_data: pandas.DataFrame
            Data containing country names, codes, and energy information.
        percentile_list: list
            List of percentiles.
        per_capita_energy_bdry_country: list of lists
            Per capita energy use by population percentile for each country.
        cum_energy_bdry_country: list of lists
            Cumulative energy use by population percentile for each country.
        filename_prefix: str
            Name used in the output file names.
        verbose_level: int
            Level of verbosity for printing progress and results.

    Returns:
        None
    """
    # Step 1: Prepare per capita energy use data
    country_info = input_data.iloc[1:, [1, 2]].values  # Extract country names and codes
    out_pc = [["per capita energy use by percentile", "kW"] + percentile_list] + \
             [list(country_info[idx]) + per_capita_energy_bdry_country[idx].tolist()
              for idx in range(len(per_capita_energy_bdry_country))]

    # Step 2: Prepare cumulative energy use data
    out_ce = [["cumulative energy use by percentile", "kW"] + percentile_list] + \
             [list(country_info[idx]) + cum_energy_bdry_country[idx].tolist()
              for idx in range(len(cum_energy_bdry_country))]

    # Step 3: Export to Excel
    out_pc_file = f"./{filename_prefix}/{filename_prefix}_country_popPct_percapEnergy.xlsx"
    out_ce_file = f"./{filename_prefix}/{filename_prefix}_country_popPct_cumEnergy.xlsx"
    
    pd.DataFrame(out_pc).to_excel(out_pc_file, index=False, header=False)
    if verbose_level > 0:
        print(f"Exported {out_pc_file}")

    pd.DataFrame(out_ce).to_excel(out_ce_file, index=False, header=False)
    if verbose_level > 0:
        print(f"Exported {out_ce_file}")

#-------------------------------------------------------------------------------------------------------------

def export_country_list(country_list, filename_prefix, verbose_level):
    """
    Exports the country list with relevant headings to an Excel file.

    Parameters:
        country_list: dictionary of lists by country index.
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
    # Step 1: Define column headings
    headings = [
        "country name",
        "country code",
        "rms fractional error of income in each World Bank bin",
        "p in Jantzen-Volper-2013 fit",
        "q in Jantzen-Volpert-2013 fit",
        "population",
        "gdp (2019$)",
        "energy (kW)",
        "income elasticity of energy use",
        "scaling factor for energy integrals",
        "income GINI coefficient",
        "energy GINI coefficient"
    ]

    # Step 2: Combine headings and country list
    # Transpose the data to match row-wise format

    # Create the pandas DataFrame
    df = pd.DataFrame(country_list)
    df.columns = headings

    # Step 3: Export to Excel
    file_name = f"./{filename_prefix}/{filename_prefix}_country_data_various.xlsx"
    df.to_excel(file_name,index=False)

    # Step 4: Print confirmation if verbose
    if verbose_level > 0:
        print(f"Exported {file_name}")

#%%
# main run

def combine_energy_data(country_list, cum_energy_bdry_country, per_capita_energy_bdry_country, global_bins_out0):
    """
    Combines energy data and produces aggregated results based on population and energy distribution.

    Parameters:
        country_list: dict
            Dictionary containing country-level data, including population information.
        cum_energy_bdry_country: numpy.array
            Cumulative energy use by population percentile for each country.
        per_capita_energy_bdry_country: numpy.ndarray
            Per capita energy use by population percentile for each country.
        global_bins_out0: int 
            Number of bins for output. If 0,  uses the input bin count.

    Returns:
        dict:
            Aggregated energy data as a dictionary with keys corresponding to different metrics.
    """
    # Step 1: Determine the number of bins
    n_bins_in = cum_energy_bdry_country.shape[1] - 1
    global_bins_out = global_bins_out0 if global_bins_out0 > 0 else n_bins_in

    # Step 2: Calculate increments (energy boundaries for each bin)
    energy_bdry_country = cum_energy_bdry_country[:, 1:] - cum_energy_bdry_country[:, :-1]

    # Step 3: Calculate population in each increment
    population_table = (np.expand_dims(country_list["Population"], axis=1) / n_bins_in).repeat(n_bins_in, axis=1)

    # Step 4: Flatten and sort the data
    sorted_table = np.hstack([
        per_capita_energy_bdry_country[:, 1:].flatten()[:, np.newaxis],
        population_table.flatten()[:, np.newaxis],
        energy_bdry_country.flatten()[:, np.newaxis]
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
    d_idx = 1.0 / global_bins_out
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

    for idx in range(global_bins_out):
        i_low = max(i_min, idx * d_idx)
        i_high = min(i_max, (idx + 1) * d_idx)
        
        frac_energy_high = np.exp(energy_fn(i_high))
        frac_energy_low = np.exp(energy_fn(i_low))
        cum_energy_in_bin = frac_energy_high - frac_energy_low
        per_capita_energy_in_bin = cum_energy_in_bin * total_energy / (total_population / global_bins_out)
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
    #export_data = [headings] + combined_data.tolist() if isinstance(combined_data, np.ndarray) else [headings] + combined_data


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
    Exports redistribution energy data with appropriate headings to an excel file
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
    pct_steps = 10000  # 100 for testing, 10000 for production
    energy_steps = 1000 # 100 for test and summary output, 1000 for production
    global_bins_out = 1000 # number of bins for output
    verbose_level = 2
    dir = r"C:\Users\kcaldeira\My Drive\energy_distribution"
    data_input_file_name = "Data-input_Virguez-et-al_2025_2025-01-29.xlsx"
    epsilon = 1e-12  # Approximately one-hundredth of a person for 10^10 people
   #--------------------------------------------------------------------------

    # Set the working directory to the project folder
    os.chdir(dir)

    # Import the Excel file
    input_data = pd.read_excel(data_input_file_name, sheet_name=0)

    # Get the dimensions of the data frame
    dimensions = input_data.shape
    print(f"Dimensions of the input data: {dimensions}")
    
    pct_dx = 1. / pct_steps
    percentile_list = (np.arange(0, 1 + pct_dx, pct_dx)).tolist()
    percentile_list[0] = epsilon
    percentile_list[-1] = 1.0 - epsilon
    n_groups = 5
    group_names = ["low", "low-middle", "middle", "middle-high", "high"]
    idx_group = 6 # 6 means energy, 5 means income

    filename_prefix = f"{run_name}_p{pct_steps}_e{energy_steps}_{datetime.now().replace(second=0, microsecond=0).isoformat().replace(':', '-').replace('T', '_')[:-3]}"
    
    # make directory for output files
    if not os.path.exists(filename_prefix):
        os.makedirs(filename_prefix)

    # Start timing
    start_time = time.time()

     # Step 0: Compute gamma (global elasticity of energy use)
    gamma_res = compute_elasticity_of_energy_use(input_data)
    gamma = gamma_res[0]
    if verbose_level > 0: print(f"Gamma: {gamma}")

    # Step 1: Prepare country-level data

    country_list = prep_country_level_data(input_data, gamma, epsilon, verbose_level)

    key_variables = create_key_variables(country_list, percentile_list, filename_prefix, energy_steps, verbose_level)
    elapsed_time = time.time() - start_time

    # Print the timing result
    print(f"Execution time: {elapsed_time:.2f} seconds")

    # Export country level data
    export_country_list(key_variables["country_list"], filename_prefix, verbose_level)

    # Export per capita and cumulative energy use by percentile for each country
    export_countries_percentile(input_data, percentile_list, key_variables["per_capita_energy_bdry_country"], key_variables["cum_energy_bdry_country"], filename_prefix, verbose_level)

    # Identify country groups based on per capita income
    country_groups = find_country_groups_per_capita(input_data, n_groups, idx_group)
    group_indices = find_country_group_indices(country_groups, key_variables["country_list"])


    # Export per capita and cumulative energy use by percentile for each group
    export_groups_percentile(group_indices, group_names, "energy", input_data, 
                    percentile_list,
                    key_variables["per_capita_energy_bdry_country"], 
                    key_variables["cum_energy_bdry_country"], 
                    filename_prefix, verbose_level)
    
    # Export group population and energy data at different energy levels
    export_groups_pc_energy(group_indices, group_names, "energy",
                            key_variables["energy_level_list"], 
                            key_variables["pop_table"], 
                            key_variables["energy_table"],
                            filename_prefix, verbose_level)
    # Combine energy data and produce aggregated results based on population and energy distribution
    combined_data = combine_energy_data(key_variables["country_list"], 
                        key_variables["cum_energy_bdry_country"],
                        key_variables["per_capita_energy_bdry_country"], 
                        global_bins_out)
    # write out combined data
    export_combined_energy_data(combined_data, filename_prefix)

    # do energy addition and redistribution calculations
    redist = redistribute(combined_data)

    # write out redistribution data
    export_redist_energy_data(redist, filename_prefix)

# usage:
# python.exe -i "c:/Users/kcaldeira/My Drive/Edgar distribution/energy_dist.py"
# %%
