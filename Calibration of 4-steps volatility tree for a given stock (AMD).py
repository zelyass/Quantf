"""
This Python script calibrates a 4-step implied volatility tree for AMD stock, based on options data 
from September 11, 2024, and calculates the price of an American put option with a strike price of $150.

Part (a): The script constructs a volatility tree using the implied volatility function.
Part (b): It calculates the American put option price using the calibrated stock price tree 
         and optimal exercise policy.

The implied volatility function is based on the following polynomial:
σ_imp(K) = 2.772 − 0.03797 · K + 0.0002019 · K^2 − (3.418 · 10^−7) · K^3

Stock price at time 0: S_0 = 149.86
Risk-free rate: r = 0.03
Time to maturity (T): 37 days (T = 37/365)

"""


###############################################################################
#                          PART         A
###############################################################################


import numpy as np
from scipy.stats import norm

n_prime = 4
t_prime = 37/365
delta_t = t_prime/n_prime

# Volatility surface function
def implied_volatility(K):
    vol = 2.772 - 0.03797*K + 0.0002019*K**2 - (3.418 * 10**(-7))*K**3
    return vol * np.sqrt(delta_t)

# Define R as a regular function
def R(n, j, r):
    return np.exp(r*delta_t)

# Function to calculate the Black-Scholes price for a European put/call
def black_scholes(S, K, T, r, sigma, option_type="put"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    elif option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

# Function to compute Arrow-Debreu prices
def arrow_debreu_prob(p, R, lambda_prev):
    return p / R * lambda_prev

def construct_volatility_tree(S0=149.86 , r=0.03, steps=4):
    # Initialize stock prices, option prices, Arrow-Debreu prices, and probabilities
    stock_tree = np.ones((steps + 1, steps + 1))*0  # Stock prices
    lambda_tree = np.ones((steps + 1, steps + 1))*0 # Arrow-Debreu prices
    p_tree = np.ones((steps + 1, steps + 1))*0     # Risk-neutral probabilities
    
    # Set initial stock price and Arrow-Debreu price
    stock_tree[0, 0] = S0
    lambda_tree[0, 0] = 1  # Initial Arrow-Debreu price
    
    # Compute the steps including n = 1
    for n in range(1, steps + 1):
        if n == 1:
            # For n = 1, it is case 3
            K = stock_tree[n-1, 0]
            sigma_imp = implied_volatility(K)
            put_price_0 = black_scholes(stock_tree[n-1, 0], K, n*delta_t, r, sigma_imp, "put")
    
            u_0 = (stock_tree[n-1, 0] + put_price_0) / (stock_tree[n-1, 0] / R(0, 0, r) - put_price_0)
            d_0 = 1 / u_0
    
            stock_tree[1, 1] = u_0 * stock_tree[n-1, 0]  # S(1,1)
            stock_tree[1, 0] = d_0 * stock_tree[n-1, 0]  # S(1,0)
            
            if stock_tree[1, 0]>stock_tree[0, 0]*R(0,0,r):
                stocktree[1,0] = stock_tree[1, 1]*stock_tree[0, 0] / stock_tree[0, 0]
            else:
                if stock_tree[1, 1]<stock_tree[0, 0]*R(0,0,r):
                    stocktree[1,1] = stock_tree[1, 0]*stock_tree[0, 0] / stock_tree[0, 1]
                if stock_tree[1, 0]<stock_tree[0, 0]*R(0,0,r)< stock_tree[1, 1]:
                    print("No adjustments necessary for n=1")

            
    
            p_00 = (R(0, 0, r) - d_0) / (u_0 - d_0)
            p_tree[0, 0] = p_00  # Store probability for S(1,1)
    
            lambda_tree[1, 1] = arrow_debreu_prob(p_00, R(0, 0, r), lambda_tree[0, 0])
            lambda_tree[1, 0] = arrow_debreu_prob(1 - p_00, R(0, 0, r), lambda_tree[0, 0])
            
        
        elif n % 2 == 0:
            print(f"Processing even n = {n}")
            # Process j = n // 2 first -> known
            j = n // 2
            stock_tree[n, j] = stock_tree[0, 0]
            print(f"  j = {j}, stock_tree[n, j] set to {stock_tree[n, j]}")
            
            # Process j >= (n // 2 + 1) and j <= n -> Case 2
            for j in range(n // 2 + 1, n + 1):
                if j > 0:  # Ensure j-1 is a valid index
                    # Set K(n) and calculate ado[n,j]
                    Kn = stock_tree[n-1, j-1]
                    sigma_imp = implied_volatility(Kn)
                    print(f"  Calculating for j = {j}, Kn = {Kn}, sigma_imp = {sigma_imp}")
                    # Calculate Arrow-Debreu price using Black-Scholes and lambda_tree
                    if j == n:
                        ado_d = black_scholes(S0, Kn, n*delta_t, r, sigma_imp, "call") / lambda_tree[n-1, j-1]
                        print(f"    ado_d (for j == n): {ado_d}")
                    else:
                        sum_c = lambda_tree[n-1, j] / R(n-1, j, r) * (stock_tree[n-1, j] * R(n-1, j, r) - stock_tree[n-1, j-1])
                        ado_d = (black_scholes(S0, Kn, n*delta_t, r, sigma_imp, "call") - sum_c) / lambda_tree[n-1, j-1]
                        print(f"    sum_c: {sum_c}, ado_d (for j != n): {ado_d}")
                    
                    R_n1_j_1 = R(n-1, j-1, r)
                    
                    # Compute stock price S(n, j) using ado and previous stock prices
                    if R_n1_j_1 != 0:
                        dumnum = ado_d * stock_tree[n, j-1] + stock_tree[n-1, j-1] * (stock_tree[n, j-1] / R_n1_j_1 - stock_tree[n-1, j-1])
                        dumden = ado_d + stock_tree[n, j-1] / R_n1_j_1 - stock_tree[n-1, j-1]
                        print(f"    dumnum: {dumnum}, dumden: {dumden}")
                        if dumden != 0:
                            stock_tree[n, j] = dumnum / dumden
                            print(f"    stock_tree[{n}, {j}] updated to: {stock_tree[n, j]}")
                        
            # Process j < (n // 2) last -> Case 1
            for j in range(n // 2 -1 , -1 , -1):
                Kn = stock_tree[n-1, j]
                sigma_imp = implied_volatility(Kn)
                print(f"  Processing j < n // 2: j = {j}, Kn = {Kn}, sigma_imp = {sigma_imp}")
                if j == 0:
                    ado = black_scholes(S0, Kn, n*delta_t, r, sigma_imp, "put") / lambda_tree[n-1, j]
                    print(f"    ado (for j == 0): {ado}")
                else:
                    print(f"lambda_tree[{n-1}, {j-1}] = {lambda_tree[n-1, j-1]}")
                    print(f"R({n-1}, {j-1}, {r}) = {R(n-1, j-1, r)}")
                    print(f"stock_tree[{n-1}, {j}] = {stock_tree[n-1, j]}")
                    print(f"stock_tree[{n-1}, {j-1}] = {stock_tree[n-1, j-1]}")
                    sum_p = lambda_tree[n-1, j-1] / R(n-1, j-1, r) * ( stock_tree[n-1, j] - stock_tree[n-1, j-1]* R(n-1, j-1, r))
                    ado = (black_scholes(S0, Kn, n*delta_t, r, sigma_imp, "put") - sum_p) / lambda_tree[n-1, j]
                    print(f"    sum_p: {sum_p}, ado (for j != 0): {ado}")
                    
                R_n1_j = R(n-1, j, r)
                if j + 1 < n + 1:  # Ensure j+1 is a valid index
                    stock_next = stock_tree[n, j+1]
                    stock_prev_n1 = stock_tree[n-1, j]
                    dumnum = ado * stock_next + stock_prev_n1 * (stock_prev_n1 - stock_next / R_n1_j)
                    dumden = ado + stock_prev_n1 - stock_next / R_n1_j
                    print(f"    dumnum: {dumnum}, dumden: {dumden}")
                    if dumden != 0:
                        stock_tree[n, j] = dumnum / dumden
                        print(f"stock_tree[{n}, {j}] updated to: {stock_tree[n, j]}")
            
            for j in range(1,n):
              if stock_tree[n, j]<stock_tree[n-1, j]*R(n-1,j,r)< stock_tree[n, j+1]:
                  print(f"No adjustments made for n={n} , j={j},{j-1},{j+1}")
              else:
                  if stock_tree[n,j]>stock_tree[n-1,j]*R(n-1,j,r):
                      stock_tree[n,j]=stock_tree[n,j+1]*stock_tree[n-1,j]/stock_tree[n-1,j+1]
                  if stock_tree[n,j+1]<stock_tree[n-1,j]*R(n-1,j,r):
                      stock_tree[n,j+1]=stock_tree[n,j]*stock_tree[n-1,j]/stock_tree[n-1,j-1]
            
            
            for j in range(1, n+1):
              p_tree[n-1, j-1] = (R(n-1, j-1, r) * stock_tree[n-1, j-1] - stock_tree[n, j-1]) / (stock_tree[n, j] - stock_tree[n, j-1])
            
            for j in range(0, n+1):
              
              if j == 0:
                lambda_tree[n, j] = (1 - p_tree[n-1, j]) / R(n-1, j, r) * lambda_tree[n-1, j]
              elif j == n:
                lambda_tree[n, j] = p_tree[n-1, j-1] / R(n-1, j-1, r) * lambda_tree[n-1, j-1]
              else:
                lambda_tree[n, j] = (1 - p_tree[n-1, j]) / R(n-1, j, r) * lambda_tree[n-1, j] + p_tree[n-1, j-1] / R(n-1, j-1, r) * lambda_tree[n-1, j-1]
                
        elif n % 2 == 1:
            print(f"  Odd step n={n}")
            
            # Process j = (n-1) // 2 + 1 first -> Case 3
            
            j = (n-1) // 2 + 1
            print(f"    Processing j={j}")
            
            K = stock_tree[n-1, j-1]
            sigma_imp = implied_volatility(K)
            put_price_0 = black_scholes(stock_tree[0, 0], K, n*delta_t, r, sigma_imp, "put")
            print(f"    K: {K}, sigma_imp: {sigma_imp}, put_price_0: {put_price_0}")
            
            Sum_p = lambda_tree[n-1, j-2] * 1 / R(n-1, j-2, r) * (stock_tree[n-1, j-1] - stock_tree[n-1, j-2] * R(n-1, j-2, r))
            print(f"    Sum_p: {Sum_p}")
            
            V_put_n_min1__jmin1 = (put_price_0 - Sum_p) / lambda_tree[n-1, j-1]
            print(f"    V_put_n_min1__jmin1: {V_put_n_min1__jmin1}")
            
            u_n_min1__jmin1 = (stock_tree[n-1, j-1] + V_put_n_min1__jmin1) / (stock_tree[n-1, j-1] / R(n-1, j-1, r) - V_put_n_min1__jmin1)
            print(f"    u_n_min1__jmin1: {u_n_min1__jmin1}")
            
            stock_tree[n, j] = u_n_min1__jmin1 * stock_tree[n-1, j-1]
            stock_tree[n, j-1] = stock_tree[n-1, j-1] / u_n_min1__jmin1
            print(f"    Stock Tree Updated at [{n}, {j}]: {stock_tree[n, j]}")
            print(f"    Stock Tree Updated at [{n}, {j-1}]: {stock_tree[n, j-1]}")
            
            for j in range((n-1) // 2 + 2, n + 1):
               Kn = stock_tree[n-1, j-1]
               sigma_imp = implied_volatility(Kn)
     
               print(f"Debugging n={n}, j={j}")
               print(f"Kn (stock_tree[{n-1}, {j-1}]): {Kn}")
               print(f"Implied volatility for Kn: {sigma_imp}")
     
               if j == n:
                 ado_d = black_scholes(S0, Kn, n*delta_t, r, sigma_imp, "call") / lambda_tree[n-1, j-1]
                 print(f"ado_d (for j == n): {ado_d}")
               else:
                 sum_c = lambda_tree[n-1, j] / R(n-1, j, r) * (stock_tree[n-1, j] * R(n-1, j, r) - stock_tree[n-1, j-1])
                 print(f"sum_c: {sum_c}")
                 ado_d = (black_scholes(S0, Kn, n*delta_t, r, sigma_imp, "call") / lambda_tree[n-1, j] - sum_c) / lambda_tree[n-1, j-1]
                 print(f"ado_d (for j != n): {ado_d}")
     
               dumnum = ado_d * stock_tree[n, j-1] + stock_tree[n-1, j-1] * (stock_tree[n, j-1] / R(n-1, j-1, r) - stock_tree[n-1, j-1])
               dumden = ado_d + stock_tree[n, j-1] / R(n-1, j-1, r) - stock_tree[n-1, j-1]
     
               print(f"dumnum: {dumnum}")
               print(f"dumden: {dumden}")
     
               if dumden != 0:
                 stock_tree[n, j] = dumnum / dumden
                 print(f"stock_tree[{n}, {j}] updated to: {stock_tree[n, j]}")
               else:
                 print(f"dumden is zero, stock_tree[{n}, {j}] not updated.") 
            
            for j in range((n-1)//2 -1, -1, -1):
              Kn = stock_tree[n-1, j]
              sigma_imp = implied_volatility(Kn)

              print(f"Debugging n={n}, j={j}")
              print(f"Kn (stock_tree[{n-1}, {j}]): {Kn}")
              print(f"Implied volatility for Kn: {sigma_imp}")
    
              if j == 0:
                if lambda_tree[n-1, j] != 0:  # Ensure no division by zero
                  ado_d = black_scholes(S0, Kn, n*delta_t, r, sigma_imp, "put") / lambda_tree[n-1, j]
                  print(f"ado_d (for j == 0): {ado_d}")
              else:
                if lambda_tree[n-1, j-1] != 0 and lambda_tree[n-1, j] != 0:  # Ensure no division by zero
                  sum_p = lambda_tree[n-1, j-1] / R(n-1, j-1, r) * (stock_tree[n-1, j] - stock_tree[n-1, j-1] * R(n-1, j-1, r))
                  ado_d = (black_scholes(S0, Kn, n*delta_t, r, sigma_imp, "put") - sum_p) / lambda_tree[n-1, j]
                  print(f"sum_p: {sum_p}")
                  print(f"ado_d (for j != 0): {ado_d}")

              # Check bounds for j+1
              if j + 1 < stock_tree.shape[1]:
                dumnum = ado_d * stock_tree[n, j+1] + stock_tree[n-1, j] * (stock_tree[n-1, j] - stock_tree[n, j+1] / R(n-1, j, r))
                dumden = ado_d + stock_tree[n-1, j] - stock_tree[n, j+1] / R(n-1, j, r)

                print(f"dumnum: {dumnum}")
                print(f"dumden: {dumden}")
                stock_tree[n, j] = dumnum / dumden
                print(f"    Stock Tree Updated at [{n}, {j}]: {stock_tree[n, j]}")
            
            for j in range(1,n):
                if stock_tree[n, j]<stock_tree[n-1, j]*R(n-1,j,r)< stock_tree[n, j+1]:
                    print(f"No adjustments made for n={n} , j={j},{j-1},{j+1}")
                else:
                    if stock_tree[n,j]>stock_tree[n-1,j]*R(n-1,j,r):
                        stock_tree[n,j]=stock_tree[n,j+1]*stock_tree[n-1,j]/stock_tree[n-1,j+1]
                    if stock_tree[n,j+1]<stock_tree[n-1,j]*R(n-1,j,r):
                        stock_tree[n,j+1]=stock_tree[n,j]*stock_tree[n-1,j]/stock_tree[n-1,j-1]
            
            for j in range(1, n+1):
              p_tree[n-1, j-1] = (R(n-1, j-1, r) * stock_tree[n-1, j-1] - stock_tree[n, j-1]) / (stock_tree[n, j] - stock_tree[n, j-1])
            
            for j in range(0, n+1):
              if j == 0:
                lambda_tree[n, j] = (1 - p_tree[n-1, j]) / R(n-1, j, r) * lambda_tree[n-1, j]
              elif j == n:
                lambda_tree[n, j] = p_tree[n-1, j-1] / R(n-1, j-1, r) * lambda_tree[n-1, j-1]
              else:
                lambda_tree[n, j] = (1 - p_tree[n-1, j]) / R(n-1, j, r) * lambda_tree[n-1, j] + p_tree[n-1, j-1] / R(n-1, j-1, r) * lambda_tree[n-1, j-1]
                
    return stock_tree, lambda_tree, p_tree

# Run the volatility tree construction
stock_tree, lambda_tree, p_tree = construct_volatility_tree()

# Display the results
print("\nStock Prices Tree:")
print(stock_tree)
print("\nArrow-Debreu Prices Tree:")
print(lambda_tree)
print("\nProbability Tree:")
print(p_tree)

###############################################################################
#                          PART         B
###############################################################################
def american_put(K=150, r=0.03, steps=4):
    # Initialize the option price tree (a_put_tree)
    a_put_tree = np.zeros((steps + 1, steps + 1))  # Option prices

    # Iterate backward from maturity to step 0
    for n in range(steps, -1, -1):
        if n == steps:
            # Calculate option prices at maturity
            for j in range(n + 1):
                a_put_tree[n, j] = max(K - stock_tree[n, j], 0)
                if a_put_tree[n, j] == K - stock_tree[n, j]:
                    print(f"\na_put_tree[{n}, {j}] = K - stock_tree[{n}, {j}] = {a_put_tree[n, j]}")
        else:
            # Calculate option prices at earlier steps
            for j in range(n + 1):
                expected_value = 1 / R(n, j, r) * (p_tree[n, j] * a_put_tree[n + 1, j + 1] + (1 - p_tree[n, j]) * a_put_tree[n + 1, j])
                a_put_tree[n, j] = max(K - stock_tree[n, j], expected_value)
                if a_put_tree[n, j] == K - stock_tree[n, j]:
                    print(f"a_put_tree[{n}, {j}] = K - stock_tree[{n}, {j}] = {a_put_tree[n, j]}")

    # Return the option price tree
    return a_put_tree

# Assuming stock_tree and p_tree are already defined in your environment

# Call the function to get the result
result = american_put()

# Print the result
print("\nAmerican Put :")
print(result)
