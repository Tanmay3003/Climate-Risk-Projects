import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(123)

#Section 1 Portfolio setup

portfolio = [
    {"asset_id": "A001", "name": "Mumbai Port", 
     "exposure": 8_500_000, "annual_flood_prob": 0.08, "alpha": 1.2},
    {"asset_id": "A002", "name": "Chennai Warehouse", 
     "exposure": 3_200_000, "annual_flood_prob": 0.04, "alpha": 1.5},
    {"asset_id": "A003", "name": "Delhi Office", 
     "exposure": 1_500_000, "annual_flood_prob": 0.01, "alpha": 1.8},
    {"asset_id": "A004", "name": "Kolkata Factory", 
     "exposure": 5_700_000, "annual_flood_prob": 0.06, "alpha": 1.3},
    {"asset_id": "A005", "name": "Pune Data Center", 
     "exposure": 2_900_000, "annual_flood_prob": 0.02, "alpha": 1.6},
]

portfolio_df = pd.DataFrame(portfolio)
num_years = 1000
num_asset = len(portfolio_df)

#Section 2 Indepdent Asset Simulation

def simulate_asset(row,num_years):
    exposure = row["exposure"]
    flood_prob = row["annual_flood_prob"]
    alpha = row["alpha"]
    
    event_occurred = np.random.random(num_years) < flood_prob
    
    raw = exposure * (np.random.random(num_years) ** (-1/alpha)-1)
    
    capped = np.minimum(raw,exposure)
    
    losses = np.where(event_occurred,capped,0.0)
    
    return losses

def simulate_portfolio(portfolio_df,num_years):
    portfolio_matrix = portfolio_df.apply(
    lambda row: simulate_asset(row,num_years),
    axis = 1
    )
    
    portfolio_matrix = np.array(portfolio_matrix.to_list())
    
    return portfolio_matrix

loss_matrix = simulate_portfolio(portfolio_df,num_years)

# SECTION 3 — Independent Portfolio Risk Data

portfolio_total_loss = np.sum(loss_matrix,axis = 0)
AAL = np.mean(portfolio_total_loss)
portfolio_df["aal"] = np.mean(loss_matrix,axis = 1)
VaR = np.percentile(portfolio_total_loss,95)
CVaR = np.mean(portfolio_total_loss[portfolio_total_loss >= VaR])
worst_year = np.argmax(portfolio_total_loss)
worst_loss = portfolio_total_loss[worst_year]

# SECTION 4 — Independent Portfolio Results

display_df = portfolio_df[["name", "exposure", "annual_flood_prob", "aal"]].copy()
display_df["exposure"] = display_df["exposure"].apply(lambda x: f"₹{x:,.0f}" )
display_df["aal"] = display_df["aal"].apply(lambda x: f"₹{x:,.2f}")
print(display_df.to_string(index = False))

print(f"\nTotal AAL: ₹{AAL:,.2f}")
print(f"95% VaR: ₹{VaR:,.2f}")
print(f"95% CVaR: ₹{CVaR:,.2f}")

top_asset = portfolio_df.loc[portfolio_df["aal"].idxmax(), "name"]

print(f"Worst year: Year {worst_year + 1} with ₹{worst_loss:,.2f} total portfolio loss")
print(f"{top_asset} contributes the most to portfolio AAL.")

# SECTION 5 — Correlation Matrix

corr_matrix = np.array([
    [1.0, 0.3, 0.1, 0.6, 0.2],   # Mumbai Port
    [0.3, 1.0, 0.1, 0.3, 0.2],   # Chennai Warehouse
    [0.1, 0.1, 1.0, 0.1, 0.1],   # Delhi Office (low correlation)
    [0.6, 0.3, 0.1, 1.0, 0.2],   # Kolkata Factory (high with Mumbai)
    [0.2, 0.2, 0.1, 0.2, 1.0],   # Pune Data Center
])

# SECTION 6 — Correlated Simulation

def simulate_correlated_portfolio(portfolio_df,corr_matrix,num_years):
    raw = np.random.multivariate_normal(
    np.zeros(5),cov = corr_matrix,size = num_years
    )
    uniform_draws = norm.cdf(raw)
    
    flood_prob = portfolio_df["annual_flood_prob"].values
    exposure = portfolio_df["exposure"].values
    alpha = portfolio_df["alpha"].values
    
    events = uniform_draws < flood_prob
    
    random_draws = np.random.random((num_years,len(portfolio_df)))
    
    severity = exposure * (random_draws ** (-1/alpha) - 1)
    
    capped = np.minimum(severity,exposure)
    
    losses = np.where(events,capped,0.0)
    
    return losses.T

correlated_matrix = simulate_correlated_portfolio(portfolio_df,corr_matrix,num_years)

# SECTION 7 — Compare and Visualize

independent_losses = np.sum(loss_matrix,axis = 0)
correlated_losses = np.sum(correlated_matrix, axis = 0)

corr_var95 = np.percentile(correlated_losses,95)
var_diff = corr_var95 - VaR
print(f"Independent 95th% VaR = ₹{VaR:,.2f}")
print(f"Correlated 95th% VaR = ₹{corr_var95:,.2f}")
print(f"Var difference = ₹{var_diff:,.2f}")

plt.hist(
    independent_losses,
    bins = 40,
    color = "blue",
    alpha = 0.5,
    label = "Independent"
)

plt.hist(
    correlated_losses,
    bins = 40,
    color = "red",
    alpha = 0.5,
    label = "Correlated"
)

plt.axvline(
    VaR,
    color = "blue",
    linestyle = "--"
)
plt.text(VaR, plt.ylim()[1]*0.9, f'₹{VaR:,.0f}', color='blue', fontsize=8)

plt.axvline(
    corr_var95,
    color = "red",
    linestyle = "--"
)
plt.text(VaR, plt.ylim()[1]*0.7, f'₹{corr_var95:,.0f}', color='red', fontsize=8)

plt.title("Independent vs Correlated Losses")
plt.xlabel("Annual Portfolio Loss")
plt.ylabel("Frequency")

plt.legend()

plt.savefig("portfolio_risk_comparison.png")

plt.show()

# SECTION 8 — EP Curve Visualization

def compute_ep_curve(loss_array,return_period,num_years):
    sorted_array = np.sort(loss_array)
    ep_points = []
    for rp in return_periods:
        pos = int(num_years - (num_years/rp))
        pos = min(pos,num_years - 1)
        ep_points.append(sorted_array[pos])
    return ep_points
return_periods = [10, 20, 50, 100, 200, 500, 1000]
ep_independent = compute_ep_curve(independent_losses, return_periods, num_years)
ep_correlated  = compute_ep_curve(correlated_losses, return_periods, num_years)

plt.figure()
plt.plot(return_periods, ep_independent, 'b-o', label='Independent')
plt.plot(return_periods, ep_correlated,  'r-o', label='Correlated')
plt.xlabel('Return Period (years)')
plt.ylabel('Portfolio Loss (₹)')
plt.title('Exceedance Probability Curve — Independent vs Correlated')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ep_curve.png')
plt.show()
