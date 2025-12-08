import os  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from IPython.display import display

from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    Param,
    NonNegativeReals,
    Objective,
    Constraint,
    maximize,
)
from pyomo.opt import SolverFactory, TerminationCondition


"""# Function 1: Fetching and Analyzing Stock Returns"""


def fetch_and_analyze_returns(startdate, enddate, ticker_list):
    # some useful modules (these are technically redundant but harmless)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import yfinance as yf
    from IPython.display import display

    start = startdate
    end = enddate

    dow_prices = {}
    for t in ticker_list:
        try:
            df = yf.download(
                t,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if not df.empty:
                dow_prices[t] = df
            else:
                print(f"Warning: no data returned for {t}")
        except Exception as e:
            print(f"Failed {t}: {e}")

    # Build price dataframe
    prep_data = (
        pd.DataFrame(dow_prices[ticker_list[0]]["Adj Close"])
        .rename(columns={"Adj Close": ticker_list[0]})
    )

    for i in ticker_list[1:]:
        prep_data[i] = pd.DataFrame(dow_prices[i]["Adj Close"])

    # Simple returns
    return_data = pd.DataFrame()
    for i in ticker_list:
        return_data[i] = prep_data[i].pct_change()
    return_data.dropna(inplace=True)

    # Cumulative returns
    cumulative_returns = (1 + return_data).cumprod() - 1

    # Plot cumulative returns
    cumulative_returns.plot(figsize=(15, 10))
    plt.title("Cumulative Percentage Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()

    # Log returns
    log_return_data = pd.DataFrame()
    for i in ticker_list:
        log_return_data[i] = np.log(prep_data[i] / prep_data[i].shift(1))
    log_return_data.dropna(inplace=True)

    print("Simple Returns (using pct_change):")
    display(return_data.head())

    print("\nLog Returns:")
    display(log_return_data.head())

    # Plot simple returns (many subplots)
    return_data.plot(
        subplots=True,
        grid=True,
        layout=(20, 16),
        figsize=(15, 15),
    )
    plt.show()

    # Monthly returns
    monthly_returns = prep_data.resample("ME").ffill().pct_change()
    monthly_returns.dropna(inplace=True)

    print("Here are the Monthly Returns:")
    display(monthly_returns)

    # A couple of quick plots (they show inline in Colab)
    monthly_returns[ticker_list[0]].plot()
    plt.show()

    monthly_returns.plot()
    plt.show()

    # Covariance matrix
    cov_matrix = monthly_returns.cov()
    plt.figure(figsize=(20, 16))
    sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".4f", center=0)
    plt.title("Covariance Matrix of Monthly Returns")
    plt.show()

    # Correlation matrix
    cor_matrix = monthly_returns.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(cor_matrix, annot=True, cmap="coolwarm", fmt=".4f", center=0)
    plt.title("Correlation Matrix of Monthly Returns")
    plt.show()

     print("Shape of monthly_returns:", monthly_returns.shape)
    #Added summary for clarity and debugging (UA)
     print(f"[INFO] Loaded {monthly_returns.shape[0]} months of data across {monthly_returns.shape[1]} tickers.")

     return monthly_returns



"""# Function 2: Run Portfolio Model
"""


def run_portfolio_model(df: "pd.DataFrame", ipopt_executable: str = "./bin/ipopt"):
    # Concrete Model
    print("Lets define the concrete model\n\n")

    m = ConcreteModel()

    print("Lets define the decision variables now\n\n")

    # Asset list
    assets = df.columns.tolist()
    m.Assets = Set(initialize=assets)

    # Decision variables: weights
    m.x = Var(m.Assets, within=NonNegativeReals, bounds=(0, 1))

      # Extra constraint: no single asset can be more than 20% of the portfolio
    max_weight = 0.20

    def max_weight_rule(m, a):
        return m.x[a] <= max_weight

    m.max_weight = Constraint(m.Assets, rule=max_weight_rule)
    print("Constraint 4: Max weight per asset (20%) has been applied.")

    # Covariance matrix (Sigma)
    cov_df = df.cov()
    cov_dict = {(i, j): cov_df.loc[i, j] for i in assets for j in assets}
    m.Sigma = Param(m.Assets, m.Assets, initialize=cov_dict)
    print("Checking the covariance values\n\n")
    m.pprint()  # Added pprint for model structure clarity
    print(f"[INFO] Model initialized with {len(assets)} assets.")



    # Average returns
    avg_returns = df.mean().to_dict()
    m.mu = Param(m.Assets, initialize=avg_returns)

    # Objective: maximize expected return
    def total_return(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)

    m.objective = Objective(rule=total_return, sense=maximize)
    print("Printing the objective function equation\n\n")
    m.pprint()

    # Constraint 1: weights sum to 1
    print("Constraint 1: Sum to 1\n")

    def budget_constraint(m):
        return sum(m.x[a] for a in m.Assets) == 1

    m.budget = Constraint(rule=budget_constraint)

    # Constraint 2: dummy non-zero (basically redundant, but okay)
    print("Constraint 2: Must be non-zero \n\n")

    def dummy_total_risk_rule(m):
        return sum(m.x[a] for a in m.Assets) >= 0.0

    m.total_risk = Constraint(rule=dummy_total_risk_rule)

    print("Constraint 3: Sum weighted cov matrix \n\n")

    # Risk thresholds
    max_risk = 0.1
    risk_limits = np.arange(0.005, max_risk, 0.001)

    param_analysis = {}
    returns = {}

    solver = SolverFactory("ipopt", executable=ipopt_executable)

    for r in risk_limits:
        # Remove old variance constraint if it exists
        if hasattr(m, "variance_constraint"):
            m.del_component(m.variance_constraint)

        # New variance constraint
        def variance_constraint_rule(m):
            return (
                sum(
                    m.Sigma[i, j] * m.x[i] * m.x[j]
                    for i in m.Assets
                    for j in m.Assets
                )
                <= r
            )

        m.variance_constraint = Constraint(rule=variance_constraint_rule)

        result = solver.solve(m)

        if result.solver.termination_condition == TerminationCondition.infeasible:
            continue

        param_analysis[r] = [m.x[a]() for a in m.Assets]
        returns[r] = sum(m.mu[a] * m.x[a]() for a in m.Assets)

    # Efficient frontier
    df_results = pd.DataFrame(
        {
            "Risk": list(returns.keys()),
            "Return": list(returns.values()),
        }
    ).sort_values(by="Risk")

    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Risk"], df_results["Return"], marker="o", linestyle="-")
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    output_dir = os.environ.get("OUTPUT_DIR", "./output_dir/")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "efficient_frontier.png"), bbox_inches="tight")
    print("Saved efficient frontier plot.")
    plt.show()

    # Allocations per risk level
    df_allocations = pd.DataFrame(param_analysis).T
    df_allocations.columns = assets
    df_allocations["Risk"] = df_allocations.index

    plt.figure(figsize=(10, 6))
    for asset in assets:
        plt.plot(df_allocations["Risk"], df_allocations[asset], label=asset, marker="o")

    plt.title("Asset Allocation as a Function of Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.legend(title="Asset")
    plt.grid(True)
    plt.tight_layout()

    output_dir = os.environ.get("OUTPUT_DIR", "./output_dir/")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "allocations_spaghetti.png"), bbox_inches="tight")
    print("Saved allocation spaghetti plot.")
    plt.show()

    # Return numeric outputs as before
    return df_results, df_allocations


def run_portfolio_pipeline(
    ipopt_executable: str,
    tickers,
    start_date,
    end_date,
    min_months_required: int = 6,
    #Adding argument for output directory
     output_dir=None,
):

    if output_dir is None:
        output_dir = os.environ.get("OUTPUT_DIR", "./output_dir/")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Get monthly returns
    monthly_returns = fetch_and_analyze_returns(start_date, end_date, tickers)

    if monthly_returns is None or monthly_returns.empty:
        raise RuntimeError(
            "No monthly returns were produced. Check tickers/date range."
        )

    if monthly_returns.shape[0] < min_months_required:
        print(
            f"Warning: only {monthly_returns.shape[0]} monthly observations "
            f"(min recommended = {min_months_required}). Proceeding anyway."
        )

    

    # 2) Run optimization model
    df_frontier, df_allocations = run_portfolio_model(
        monthly_returns,
        ipopt_executable=ipopt_executable,
    )

        # ---------------------------------------------------------
    # 3) Build three key portfolios:
    #    - Conservative: minimum risk
    #    - Aggressive: maximum return
    #    - Balanced: best Return/Risk ratio (elbow-like)
    # ---------------------------------------------------------

    # Conservative = min risk
    idx_conservative = df_frontier["Risk"].idxmin()
    risk_conservative = df_frontier.loc[idx_conservative, "Risk"]
    ret_conservative = df_frontier.loc[idx_conservative, "Return"]

    # Aggressive = max return
    idx_aggressive = df_frontier["Return"].idxmax()
    risk_aggressive = df_frontier.loc[idx_aggressive, "Risk"]
    ret_aggressive = df_frontier.loc[idx_aggressive, "Return"]

    # Balanced = max Return/Risk ratio (simple proxy for "elbow")
    df_frontier["Return_to_Risk"] = df_frontier["Return"] / df_frontier["Risk"]
    idx_balanced = df_frontier["Return_to_Risk"].idxmax()
    risk_balanced = df_frontier.loc[idx_balanced, "Risk"]
    ret_balanced = df_frontier.loc[idx_balanced, "Return"]

    # Helper to pull allocations for a given risk level
    def get_alloc_row_for_risk(target_risk):
        row = df_allocations.loc[df_allocations["Risk"] == target_risk]
        # in case of floating comparison quirks, take the first match
        return row.iloc[0]

    assets = [c for c in df_allocations.columns if c != "Risk"]

    portfolios = []

    # Conservative row
    alloc_conservative = get_alloc_row_for_risk(risk_conservative)
    row_conservative = {"Portfolio": "Conservative", "Risk": risk_conservative, "Return": ret_conservative}
    for a in assets:
        row_conservative[a] = alloc_conservative[a]
    portfolios.append(row_conservative)

    # Balanced row
    alloc_balanced = get_alloc_row_for_risk(risk_balanced)
    row_balanced = {"Portfolio": "Balanced", "Risk": risk_balanced, "Return": ret_balanced}
    for a in assets:
        row_balanced[a] = alloc_balanced[a]
    portfolios.append(row_balanced)

    # Aggressive row
    alloc_aggressive = get_alloc_row_for_risk(risk_aggressive)
    row_aggressive = {"Portfolio": "Aggressive", "Risk": risk_aggressive, "Return": ret_aggressive}
    for a in assets:
        row_aggressive[a] = alloc_aggressive[a]
    portfolios.append(row_aggressive)

    df_key_portfolios = pd.DataFrame(portfolios)

    # Save the 3 key portfolios to CSV
    df_key_portfolios.to_csv(os.path.join(output_dir, "key_portfolios.csv"), index=False)
    print(f"Saved key portfolios (Conservative/Balanced/Aggressive) to: {os.path.join(output_dir, 'key_portfolios.csv')}")


  # 4) Save key outputs to CSV in output_dir
    monthly_returns.to_csv(os.path.join(output_dir, "monthly_returns.csv"))
    df_frontier.to_csv(os.path.join(output_dir, "efficient_frontier.csv"), index=False)
    df_allocations.to_csv(os.path.join(output_dir, "allocations_by_risk.csv"), index=False)

    print(f"Saved monthly_returns to {os.path.join(output_dir, 'monthly_returns.csv')}")
    print(f"Saved efficient frontier to {os.path.join(output_dir, 'efficient_frontier.csv')}")
    print(f"Saved allocations by risk to {os.path.join(output_dir, 'allocations_by_risk.csv')}")


    # Added pipeline completion message 
    print("[INFO] Portfolio pipeline completed successfully.")
    
    # 5) Return everything
    return monthly_returns, df_frontier, df_allocations, df_key_portfolios
