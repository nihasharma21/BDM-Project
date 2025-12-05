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
