# Portfolio Pipeline

This repo contains a small, reusable **portfolio optimization pipeline** built for the stock optimization model. 

It:

1. Downloads **daily prices** from Yahoo Finance for a user-specified list of tickers.
2. Converts them into **monthly returns**.
3. Builds a **Markowitz efficient frontier** using `pyomo` + `ipopt`.
4. Generates:
   - An efficient frontier table (risk vs expected return),
   - A table of stock allocations at each risk level,
   - Plots for the efficient frontier and the “spaghetti” allocation chart.
5. Saves all key outputs (CSVs + PNGs) into `./output_dir/` so anyone can run and review the results.

---

## Repository Structure

Within the `portfolio-pipeline/` folder:

```text
portfolio-pipeline/
├─ main.py                    # CLI entrypoint
├─ requirements.txt           # Python dependencies
├─ src/
│  └─ portfolio_pipeline.py   # Core pipeline functions
├─ output_dir/                # Created at runtime; stores CSVs + plots
└─ bin/
   └─ ipopt                   # Ipopt executable (downloaded via idaes)

Note: output_dir/ is created automatically when you run the script.
All tables and images are saved there.

