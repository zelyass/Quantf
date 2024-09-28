"""
Code Description

This Python script is designed to implement a financial model involving bond pricing,
interest rate tree construction, and option pricing using the Binomial Model (BDT).
The main components of the code include:

1. Data Preparation:
   - A DataFrame is created to hold yield, bond price, and yield volatility data for
     three different bonds. The bond prices are calculated using the formula 
     Bond Price = 1 / (1 + Yield) ** index.

2. Matrix Initialization:
   - Several matrices are initialized to hold probabilities, interest rates, and Arrow-Debreu 
     (AD) prices, which are essential for martingale pricing and building the interest rate tree.

3. Martingale Pricing Function:
   - A function `martingale_pricing` is defined to calculate the Arrow-Debreu price 
     based on the node position in the tree, updating the `ado_matrix`.

4. Jamishidian’s Forward Induction:
   - The function `jfi` calculates the Arrow-Debreu prices at each node, which are needed
     to compute bond prices and interest rates.

5. Rate Volatility Calculation:
   - A function `rate_vol` computes the volatility of interest rates using the logarithmic 
     transformation.

6. Main Loop for Building Trees:
   - The main loop iterates through each time step to fill in the interest rate tree 
     and compute the corresponding yield and rate values. It solves systems of equations 
     to determine interest rates at each step, taking into account the calculated 
     Arrow-Debreu prices.
"""

import numpy as np
import pandas as pd
import sympy as sp

#We will create the data from the exercise as a DataFrame
data = pd.DataFrame(index=range(1, 4), 
                    columns=['Yield', 'Bond Price', 'Yield Vol'],
                    data=[[0.039, np.NaN, np.NaN], 
                          [0.045, np.NaN, 0.168], 
                          [0.055, np.NaN, 0.155]])

#We shall upadte the Bond Price using the formula 1/(1+Yield)^index
for i in data.index:
    yield_value = data.loc[i, 'Yield']
    data.loc[i, 'Bond Price'] = 1 / (1 + yield_value)**i

# Next we will display the updated DataFrame
print(data)

N = len(data)
p_tilda = 0.5  # probability value assumed to be 1/2 in BDT

# We will be setting up the probability, interest rates, AD, and Yield trees :

prob_matrix = np.array([[p_tilda for _ in range(N + 1)] for _ in range(N + 1)])
r_matrix = np.ones((N, N))*0
ado_matrix= np.ones((N, N,N,N))*0
r_matrix[0, 0] = data['Yield'][1]
ado_matrix[0, 0, 0, 0] = 1


# We will define the following function for martingale pricing
def martingale_pricing(i, n, j):
  #We shall Calculate Arrow-Debreu price based on the node position
    if i == j - 1:
        ado = prob_matrix[n - 1, j - 1] / (1 + r_matrix[n - 1, j - 1])
    elif i == j:
        ado = (1 - prob_matrix[n - 1, j]) / (1 + r_matrix[n - 1, j])
    else:
        ado = 0 # No price if not at a valid node
    ado_matrix[n - 1, i, n, j] = ado # we willl tore the price in the AD matrix



# We will define the following for Jamishidian’s Forward Induction
def jfi(n, i, m, j):
    if j == 0:
        ado = ado_matrix[n, i, m - 1, j] * (1 - prob_matrix[m - 1, j]) / (1 + r_matrix[m - 1, j])
    elif 0 < j < m:
        ado = (ado_matrix[n, i, m - 1, j] * (1 - prob_matrix[m - 1, j]) / (1 + r_matrix[m - 1, j]) + 
              ado_matrix[n, i, m - 1, j - 1] * prob_matrix[m - 1, j - 1] / (1 + r_matrix[m - 1, j - 1]))
    elif j == m:
        ado = ado_matrix[n, i, m - 1, j - 1] * prob_matrix[m - 1, j - 1] / (1 + r_matrix[m - 1, j - 1])
    ado_matrix[n, i, m, j] = ado

# We define the following to compute rate volatility
def rate_vol(x):
    # we should convert x to a float before applying np.log or else we will encounter an error
    return (1 / 2) * np.log(float(x))


# Now the following is the Main loop to fill the yield and rate tree
for n in range(1, N): 
    
    # We need to solve a system of two equations in two unknowns
    r = sp.Symbol('r')
    sigma = sp.Symbol('sigma') 
    
    #PART 1
    
    # Fill a level of time 0 AD tree using JFI
    for j in range(0, n + 1):
        jfi(0, 0, n, j)
        
    # we previously calculated bon prices
    bond_price = data['Bond Price'][n + 1]
    
    if n == 1:
        # At time 1 rates are the same as yields of bonds at time 1 with maturity 2
        rate_volatility = np.exp(2 * data['Yield Vol'][n + 1])
        B = 0  # Initialize the sum
        for j in range(0, n + 1):
            B += ado_matrix[0, 0, n, j] / (1 + r * rate_volatility**j)  # Accumulate the result
        rate = np.array([sp.re(x) for x in sp.solvers.solve(bond_price - B, r)])
        rate = rate[rate > 0][0]  # Set the interest rate to the positive root
    else:
        # After time one we have to construct a system to solve numerically
        B = 0  # Initialize the sum
        for j in range(0, n + 1):
            B += ado_matrix[0, 0, n, j] / (1 + r * sigma**j)  # Accumulate the result
    
    #PART 2
    
    if n > 1: 
        Y = [] 
        for i in range(0, 2):  # Yields at (1,0;n+1) and (1,1;n+1)
            p = 0  # Bond prices
            for j in range(0, n + 1):  # Payoffs at terminal nodes: (n,0), (n,1), ..., (n,n)
                # Calculate one step AD prices using risk-neutral pricing formula
                martingale_pricing(i, n, j)
                if n > 2:
                    jfi(1, i, n, j)
                p += ado_matrix[1, i, n, j] / (1 + r * sigma**j)
            Y.append(p**(-(1/n)) - 1)
            
    #Now we solve the overall Equation :)
    
    if n > 1:
        solution = sp.nsolve(
            [Y[0] * np.exp(2 * data['Yield Vol'][n + 1]) - Y[1], bond_price - B],
            [r, sigma], [0.05, 0.01])
        rate = solution[0]
        rate_volatility = solution[1]
        
    # We Substitute the values found to calculate actual time 1 yields
    if n > 1:
        for i in range(0, 2):
            p = 0  # Bond prices
            for j in range(0, n + 1):  # Payoffs at terminal nodes of time n
                p += ado_matrix[1, i, n, j] / (1 + rate * rate_volatility**j)
            
    r_matrix[n, 0] = rate  # We Save the rate to the tree
    
    # THEN Compute the rest of the interest rates applying BDT condition for volatility
    for j in range(1, n + 1):
        r_matrix[n, j] = r_matrix[n, 0] * rate_volatility**j
print("\nInterest rate Tree:")        
print(r_matrix)
