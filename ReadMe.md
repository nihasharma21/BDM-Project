#  Strategic Synergy- Stock Optimization Model 
### *Portfolio Optimization Pipeline – OPIM 5641 Final Project*

This project implements a reusable, end-to-end **portfolio optimization pipeline** for the OPIM 5641 (Business Decision Modeling) final assignment.  
It downloads real stock data, computes monthly returns, builds an efficient frontier using Modern Portfolio Theory (MPT), and saves all outputs for easy evaluation.

The entire workflow is packaged so anyone can clone the repo and run the model with a single command.

---

#  Project Overview (High-Level)

This project performs the following steps:

1. **Download real stock price data** using Yahoo Finance  
2. **Convert daily prices to monthly returns**  
3. **Build a Markowitz portfolio optimization model** using Pyomo + IPOPT  
4. **Generate visualizations**:  
   - Efficient Frontier  
   - Asset Allocation (“Spaghetti Plot”)  
5. **Save all results** (tables + charts) into `output_dir/`  
6. **Provide a reusable pipeline** that supports future extensions such as paper-trading and out-of-sample testing.

---


---

# What the Project Does (Step-By-Step)

## **1. Data Download**
The pipeline begins by downloading **daily adjusted closing prices** for any list of user-provided stock tickers using the Yahoo Finance API.  
This provides the raw financial data for modeling.

---

## **2. Convert Daily Prices to Monthly Returns**
Daily data is cleaned and resampled into **monthly percentage returns**, which are more stable and suitable for portfolio optimization.

The pipeline also computes and visualizes:
- Daily percentage returns  
- Log returns  
- Cumulative returns  
- Covariance and correlation heatmaps  

These exploratory outputs help understand risk relationships among assets.

---

## **3. Build an Optimization Model**
The core of the project is a **Modern Portfolio Theory (MPT)** optimization model implemented with Pyomo and solved using IPOPT.

The model:
- Controls one decision variable per stock (its portfolio weight)  
- Forces all weights to sum to 1 (fully invested)  
- Enforces long-only weights (0–100%)  
- Maximizes expected return  
- Iteratively explores multiple **risk levels** using the covariance matrix  

This produces the **efficient frontier**—the curve of optimal risk/return trade-offs.

### Extra Constraints Applied in Optimization

To reflect realistic portfolio construction and prevent over-concentration in any one asset, the following additional constraint was applied:

Maximum allocation per asset: 20% cap
Each stock is allowed to contribute no more than 0.2 (20%) of the total portfolio weight in any given optimized solution.

This constraint forces diversification and simulates portfolio restrictions often applied by institutional investors.

---

## **4. Produce Visualizations**
Two main charts are generated:

### **Efficient Frontier**
A plot of:
- Portfolio risk (variance)  
- Portfolio expected return  

This shows which portfolios are optimal for each risk level.

### **Asset Allocation Plot**
Also known as the “spaghetti plot,” showing:
- How weights for each stock change as risk increases  
- Which assets dominate conservative vs aggressive portfolios  

---

## **5. Save All Outputs into `output_dir/`**
The pipeline writes results to disk for easy review:

- `monthly_returns.csv`  
- `efficient_frontier.csv`  
- `allocations_by_risk.csv`  
- `efficient_frontier.png`  
- `allocations_spaghetti.png`  

---

#  How to Run the Project



# **1.  Run Everything in Google Colab** (Recommended)

This project is designed so anyone can open a Colab notebook and run the full pipeline with **one setup cell**.

### **Step 1 — Open Colab**

Create a new notebook or open the provided notebook.

### **Step 2 — Clone the repo and install dependencies**

### **Step 3 - Run the portfolio pipeline**
 Choose your ticker list
 Then run the full pipeline

 ### **Step 4 - View saved outputs**
 

---

# Output Summary

The model generates an efficient frontier of portfolios by maximizing return for a given level of risk (variance), subject to the above constraints.

## Key Outputs

### Efficient Frontier Curve (efficient_frontier.png)

X-axis: Portfolio risk (variance)
Y-axis: Expected return
The curve shows the trade-off between risk and return for optimized portfolios.
The curve flattens beyond a certain point, indicating diminishing return for increased risk under the constraints.

### Asset Allocation Across Frontier (allocations_spaghetti.png)

Each line: A stock's allocation at each portfolio along the frontier

### Allocation CSV (efficient_frontier.csv)

Provides detailed allocations per asset at each risk level.
Useful for selecting specific portfolios (e.g., conservative, balanced, aggressive) and understanding underlying holdings.

##  Observations From the Allocation-by-Risk Chart

### **Overall Allocation Behavior**
The allocation-by-risk (“spaghetti”) chart shows how the portfolio composition changes as the optimizer moves across increasing levels of portfolio risk. A clear pattern emerges:

- A small set of high-return stocks consistently receive the highest allocations.
- Many stocks receive near-zero weights across most of the frontier, indicating that they are not attractive given their return–risk–correlation profile.
- The 20% position cap is **binding** for several assets—meaning the optimizer wants to allocate more than 20% to them, but is restricted by the diversification constraint.

This produces a frontier where additional risk does not always translate into more expected return, because the model is already maxing out the most attractive assets.

---

### **Specific Insight: Microsoft (MSFT)**
One notable pattern is that **MSFT stays at the maximum allowed 20% weight across nearly the entire efficient frontier**.  

This does **not** mean “the model is putting everything into Microsoft.” Instead:

- MSFT is one of the most attractive stocks based on monthly returns and covariance.
- Because of the 20% cap, the optimizer allocates **exactly 20%** to MSFT whenever possible.
- Other stocks such as EQIX, PLTR, ADBE, META, and T exhibit similar behavior—they also hit the 20% ceiling across long portions of the frontier.

This confirms that the **max-weight diversification constraint is actively shaping the portfolio**, preventing unrealistic concentration in a few dominant names.

