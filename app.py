# -*- coding: utf-8 -*-
"""Budget Tracker Application - User-Friendly GUI Version"""

# Install dependencies for Colab (run this cell first in Colab)
# This part is not needed in the .py file as you've already installed in the notebook.
#!pip install alpha_vantage pandas numpy scikit-learn statsmodels python-dotenv streamlit matplotlib

# Import dependencies
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from dotenv import load_dotenv
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import io  # For pie chart in Streamlit

# Set environment variable for API key (replace with your key or use Colab Secrets)
# Replace os.getenv with st.secrets
try:
    api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
except (KeyError, ValueError):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'default_key')
os.environ['ALPHA_VANTAGE_API_KEY'] = api_key

# User class - Encapsulated financial details with clear methods for operations
class User:
    """Class to represent a user with financial details."""
    def __init__(self, name):
        """Initialize user with name and financial attributes."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Name must be a non-empty string.")
        self._name = name.strip()
        self._total_income = 0.0
        self._total_expenses = 0.0
        self._savings_goal = 0.0

    def add_income(self, amount):
        """Add income to the user's total income."""
        if not isinstance(amount, (int, float)) or amount <= 0:
            return False
        self._total_income += amount
        return True

    def add_expense(self, amount):
        """Add expense to the user's total expenses."""
        if not isinstance(amount, (int, float)) or amount <= 0 or amount > self._total_income:
            return False
        self._total_expenses += amount
        return True

    def update_savings_goal(self, goal):
        """Update the user's savings goal."""
        if not isinstance(goal, (int, float)) or goal < 0:
            return False
        self._savings_goal = goal
        return True

    @property
    def name(self):
        return self._name

    @property
    def total_income(self):
        return self._total_income

    @property
    def total_expenses(self):
        return self._total_expenses

    @property
    def savings_goal(self):
        return self._savings_goal

# Transaction class - Encapsulated transaction data with validation for consistency
class Transaction:
    """Class to represent a financial transaction."""
    def __init__(self, amount, category, transaction_type):
        """Initialize transaction with amount, category, and type."""
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("Amount must be positive.")
        if transaction_type not in ["income", "expense"]:
            raise ValueError("Transaction type must be 'income' or 'expense'.")
        self._amount = float(amount)
        self._category = category.strip() if isinstance(category, str) else "General"
        self._transaction_type = transaction_type
        self._date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def record_transaction(self):
        """Record the transaction (returns True if valid)."""
        return self._amount > 0

    def get_details(self):
        """Return transaction details as a string."""
        return (f"Type: {self._transaction_type}, Amount: ${self._amount:.2f}, "
                f"Category: {self._category}, Date: {self._date}")

    @property
    def amount(self):
        return self._amount

    @property
    def category(self):
        return self._category

class Database:
    def __init__(self, db_name='budget.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT,
                date TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def add_transaction(self, trans_type, amount, category, date):
        self.conn.execute('INSERT INTO transactions (type, amount, category, date) VALUES (?, ?, ?, ?)',
                          (trans_type, amount, category, date))
        self.conn.commit()

    def get_transactions(self):
        return pd.read_sql_query("SELECT * FROM transactions", self.conn)

    def etl_from_csv(self, csv_path='sample_transactions.csv'):
        try:
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df = df.dropna()
            for _, row in df.iterrows():
                self.add_transaction(row['type'], row['amount'], row['category'], row['date'])
            return "ETL completed."
        except Exception as e:
            return f"ETL error: {str(e)}"

    def ad_hoc_query(self, query="SELECT category, SUM(amount) FROM transactions WHERE type='expense' GROUP BY category"):
        return pd.read_sql_query(query, self.conn)

    def close(self):
        self.conn.close()

# BudgetManager class - Manages budget with composition of User and Transaction
class BudgetManager:
    """Class to manage the user's budget and transactions."""
    def __init__(self, user):
        """Initialize with a user and an empty transaction list."""
        if not isinstance(user, User):
            raise ValueError("User must be an instance of User class.")
        self._user = user
        self.db = Database()
        self._transactions = []  # Optional in-memory cache

    def calculate_balance(self):
        """Calculate the remaining balance."""
        return self._user.total_income - self._user.total_expenses

    def add_transaction(self, transaction):
        """Add a transaction if valid."""
        if not isinstance(transaction, Transaction) or not transaction.record_transaction():
            return False
        self.db.add_transaction(transaction._transaction_type, transaction.amount, transaction.category, transaction._date)
        self._transactions.append(transaction)  # Cache if needed
        if transaction._transaction_type == "income":
            return self._user.add_income(transaction.amount)
        return self._user.add_expense(transaction.amount)

    def check_savings_progress(self):
        """Check progress toward savings goal."""
        if self._user.savings_goal <= 0:
            return 0.0
        balance = self.calculate_balance()
        return min(balance / self._user.savings_goal * 100, 100)

    def generate_report(self):
        """Generate a summary report of transactions and balance."""
        try:
            transactions = self.db.get_transactions()
            report = f"\nBudget Report for {self._user.name}:\n"
            report += f"Total Income: ${self._user.total_income:.2f}\n"
            report += f"Total Expenses: ${self._user.total_expenses:.2f}\n"
            report += f"Remaining Balance: ${self.calculate_balance():.2f}\n"
            report += f"Savings Goal: ${self._user.savings_goal:.2f} ({self.check_savings_progress():.1f}% achieved)\n"
            report += "Transactions:\n"
            if transactions.empty:
                report += "- No transactions recorded.\n"
            else:
                for _, row in transactions.iterrows():
                    report += f"- Type: {row['type']}, Amount: ${row['amount']:.2f}, Category: {row['category']}, Date: {row['date']}\n"
            return report
        except Exception as e:
            return f"Error generating report: {str(e)}"

# ReportGenerator class - Handles reports and visualizations with clear feedback
class ReportGenerator:
    """Class to generate visual and file-based reports."""
    def __init__(self, budget_manager):
        """Initialize with a BudgetManager instance."""
        if not isinstance(budget_manager, BudgetManager):
            raise ValueError("BudgetManager must be an instance of BudgetManager class.")
        self._budget_manager = budget_manager
        self._category_totals = {}

    def update_category_totals(self):
        self._category_totals.clear()
        transactions = self._budget_manager.db.get_transactions()
        for _, row in transactions.iterrows():
            if row['type'] == "expense":
                category = row['category']
                amount = row['amount']
                self._category_totals[category] = self._category_totals.get(category, 0) + amount

    def create_pie_chart(self):
        """Create a pie chart of expense categories."""
        self.update_category_totals()
        if not self._category_totals:
            return "No expenses to visualize."
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(self._category_totals.values(), labels=self._category_totals.keys(), autopct='%1.1f%%')
        ax.set_title("Expense Distribution")
        # Save to buffer for Streamlit
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    def export_to_file(self, filename="budget_report.txt"):
        """Export the report to a text file."""
        try:
            with open(filename, 'w') as f:
                f.write(self._budget_manager.generate_report())
            return f"Report exported to {filename}"
        except Exception as e:
            return f"Error exporting report: {str(e)}"

# InvestmentPortfolio class - Uses strategy pattern for consistent risk-based allocations
class InvestmentPortfolio:
    """Class to manage a user's investment portfolio."""
    def __init__(self, user_balance, risk_tolerance="moderate"):
        """Initialize portfolio with balance and risk tolerance."""
        if not isinstance(user_balance, (int, float)) or user_balance < 0:
            raise ValueError("Balance must be non-negative.")
        if risk_tolerance.lower() not in ["low", "moderate", "high"]:
            raise ValueError("Risk tolerance must be 'low', 'moderate', or 'high'.")
        self._balance = user_balance
        self._risk_tolerance = risk_tolerance.lower()
        self._allocations = {"stocks": 0.0, "bonds": 0.0, "cash": 0.0}
        self._set_default_allocation()

    def _set_default_allocation(self):
        """Set default allocation based on risk tolerance (strategy pattern)."""
        if self._risk_tolerance == "low":
            self._allocations = {"stocks": 0.3, "bonds": 0.5, "cash": 0.2}
        elif self._risk_tolerance == "moderate":
            self._allocations = {"stocks": 0.7, "bonds": 0.2, "cash": 0.1}
        else:  # high
            self._allocations = {"stocks": 0.9, "bonds": 0.0, "cash": 0.1}

    def calculate_investment_amounts(self):
        """Calculate dollar amounts for each asset class."""
        return {asset: amount * self._balance for asset, amount in self._allocations.items()}

    def get_portfolio_summary(self):
        """Return a summary of the portfolio."""
        amounts = self.calculate_investment_amounts()
        return (f"Portfolio for ${self._balance:.2f} with {self._risk_tolerance} risk:\n"
                f"- Stocks: ${amounts['stocks']:.2f} ({self._allocations['stocks']*100:.1f}%)\n"
                f"- Bonds: ${amounts['bonds']:.2f} ({self._allocations['bonds']*100:.1f}%)\n"
                f"- Cash: ${amounts['cash']:.2f} ({self._allocations['cash']*100:.1f}%)")

# InvestmentPredictor class - ML for efficient, error-reduced predictions
class InvestmentPredictor:
    """Class to predict optimal investment allocations using supervised learning."""
    def __init__(self):
        """Initialize with a pre-trained model or train on synthetic data."""
        self._model = LinearRegression()
        self._scaler = StandardScaler()
        self._train_model()

    def _train_model(self):
        """Train the model on synthetic data."""
        try:
            np.random.seed(42)
            n_samples = 100
            balance = np.random.uniform(1000, 5000, n_samples)
            savings_rate = np.random.uniform(0.1, 0.5, n_samples)
            risk_tolerance = np.random.randint(0, 3, n_samples)  # 0:low, 1:moderate, 2:high
            goal_amount = np.random.uniform(500, 2000, n_samples)

            allocation_stocks = np.where(risk_tolerance == 2, 0.9, np.where(risk_tolerance == 1, 0.7, 0.3))
            allocation_bonds = np.where(risk_tolerance == 0, 0.5, np.where(risk_tolerance == 1, 0.2, 0.0))
            allocation_cash = 1 - allocation_stocks - allocation_bonds

            X = np.column_stack((balance, savings_rate, risk_tolerance, goal_amount))
            y = np.column_stack((allocation_stocks, allocation_bonds, allocation_cash))

            X_scaled = self._scaler.fit_transform(X)
            self._model.fit(X_scaled, y)
        except Exception as e:
            print(f"Error training ML model: {str(e)}")

    def forecast_returns(self, historical_returns):
        """Forecast next return using ARIMA time-series model."""
        try:
            df = pd.Series(historical_returns)
            if len(df) < 3:  # ARIMA needs at least a few points
                raise ValueError("Insufficient historical data for ARIMA forecast.")
            model = ARIMA(df, order=(1,1,1))  # Simple order; tune as needed
            model_fit = model.fit()
            pred = model_fit.forecast(steps=1)[0]
            return pred
        except Exception as e:
            print(f"ARIMA forecast error: {str(e)}. Using fallback 0.0.")
            return 0.0

    def predict_allocation(self, balance, savings_rate, risk_tolerance, goal_amount, historical_returns=None):
        """Predict optimal allocation based on user financial data."""
        try:
            risk_encoded = {"low": 0, "moderate": 1, "high": 2}[risk_tolerance.lower()]
            X_new = np.array([[balance, savings_rate, risk_encoded, goal_amount]])
            X_new_scaled = self._scaler.transform(X_new)
            allocations = self._model.predict(X_new_scaled)[0]
            allocations = np.maximum(allocations, 0)

            if historical_returns:
                arima_pred = self.forecast_returns(historical_returns)
                adjust_factor = max(0, min(0.1, arima_pred / 100))  # Cap adjustment
                allocations[0] += adjust_factor  # Boost stocks
                allocations[2] -= adjust_factor  # Reduce cash

            total = allocations.sum()
            if total == 0:
                allocations = np.array([1/3, 1/3, 1/3])
            else:
                allocations /= total

            return {"stocks": allocations[0], "bonds": allocations[1], "cash": allocations[2]}
        except Exception as e:
            print(f"Error predicting allocation: {str(e)}")
            return {"stocks": 0.33, "bonds": 0.33, "cash": 0.34}

    def get_prediction_summary(self, balance, savings_rate, risk_tolerance, goal_amount, historical_returns=None):
        """Return a summary of the predicted allocation."""
        try:
            alloc = self.predict_allocation(balance, savings_rate, risk_tolerance, goal_amount, historical_returns)
            return (f"Predicted Portfolio for ${balance:.2f}:\n"
                    f"- Stocks: ${balance * alloc['stocks']:.2f} ({alloc['stocks']*100:.1f}%)\n"
                    f"- Bonds: ${balance * alloc['bonds']:.2f} ({alloc['bonds']*100:.1f}%)\n"
                    f"- Cash: ${balance * alloc['cash']:.2f} ({alloc['cash']*100:.1f}%)")
        except Exception as e:
            return f"Error generating prediction summary: {str(e)}"

# InvestmentAdvisor class - Integrates real-time data with ML for effective advice
class InvestmentAdvisor:
    def __init__(self, budget_manager):
        if not isinstance(budget_manager, BudgetManager):
            raise ValueError("BudgetManager must be an instance of BudgetManager class.")
        self._budget_manager = budget_manager
        self._api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'TNCT1LHKBZC5JF6W')
        self._ts = TimeSeries(key=self._api_key, output_format='pandas')
        self._predictor = InvestmentPredictor()
        self._historical_data = None
        self._real_time_data = None
        self._avg_returns = {"stocks": 0.08, "bonds": 0.03, "cash": 0.01}
        self._volatility = {"stocks": 0.15, "bonds": 0.05, "cash": 0.01}
        self._data_status = "Using fallback data (8% return, 15% volatility for stocks)."
        self._load_data()

    def _load_data(self):
        """Load historical and attempt real-time data for AAPL."""
        try:
            # Historical data (assume file or fallback)
            try:
                self._historical_data = pd.read_csv('aapl_historical.csv')
                self._historical_data['Date'] = pd.to_datetime(self._historical_data['Date'])
                self._calculate_historical_returns()
                self._data_status = "Loaded historical AAPL data."
            except FileNotFoundError:
                print("Historical data file 'aapl_historical.csv' not found. Using fallback.")

            # Real-time data
            data, _ = self._ts.get_intraday(symbol='AAPL', interval='1min', outputsize='compact')
            self._real_time_data = data
            self._calculate_real_time_metrics()
            self._data_status = "Loaded real-time AAPL data."
        except Exception as e:
            print(f"Error fetching data: {e}. Using fallback data.")
            self._data_status = f"API error: {str(e)}. Using fallback data."

    def _calculate_historical_returns(self):
        """Calculate average annual returns and volatility from historical data."""
        if self._historical_data is not None and 'Adj Close' in self._historical_data.columns:
            df = self._historical_data.copy()
            df['Returns'] = df['Adj Close'].pct_change().dropna()
            if not df['Returns'].empty:
                self._avg_returns["stocks"] = df['Returns'].mean() * 252
                self._volatility["stocks"] = df['Returns'].std() * (252 ** 0.5)

    def _calculate_real_time_metrics(self):
        """Calculate average annual returns and volatility from the latest real-time data."""
        if self._real_time_data is None or '4. close' not in self._real_time_data.columns:
            self._data_status = "No valid real-time data available. Using fallback data."
            return

        try:
            close_prices = self._real_time_data['4. close'].dropna()
            if len(close_prices) < 2:
                self._data_status = "Insufficient real-time data points. Using fallback data."
                return

            returns = close_prices.pct_change().dropna()
            n_points = len(returns)
            if n_points == 0:
                self._data_status = "No valid returns calculated. Using fallback data."
                return

            annual_factor = 252 / (n_points / (60 * 6.5))
            annual_factor = max(1, min(252, annual_factor))

            self._avg_returns["stocks"] = returns.mean() * annual_factor
            self._volatility["stocks"] = returns.std() * (annual_factor ** 0.5)

            self._data_status = f"Calculated from {n_points} real-time data points. Annualized with factor {annual_factor:.1f}."
        except Exception as e:
            print(f"Error in real-time metrics calculation: {e}")
            self._data_status = f"Error calculating metrics: {str(e)}. Using fallback data (8% return, 15% volatility)."

    def suggest_investment(self, goal_amount, risk_tolerance="moderate", historical_returns=None):
        """Suggest an investment strategy with real-time data and ML predictions."""
        try:
            if not isinstance(goal_amount, (int, float)) or goal_amount <= 0:
                raise ValueError("Investment goal must be positive.")
            if risk_tolerance.lower() not in ["low", "moderate", "high"]:
                raise ValueError("Risk tolerance must be 'low', 'moderate', or 'high'.")

            balance = self._budget_manager.calculate_balance()
            if balance <= 0:
                return "No funds available for investment."

            invest_amount = min(balance, goal_amount)
            portfolio = InvestmentPortfolio(invest_amount, risk_tolerance)
            advice = f"Investment Recommendation (Balance: ${balance:.2f}, Goal: ${goal_amount:.2f}):\n"
            advice += portfolio.get_portfolio_summary()

            savings_rate = min(balance / goal_amount, 1.0) if goal_amount > 0 else 0.0
            ml_prediction = self._predictor.get_prediction_summary(balance, savings_rate, risk_tolerance, goal_amount, historical_returns)
            advice += f"\nML-Predicted Strategy:\n{ml_prediction}"

            if balance < goal_amount:
                advice += "\nWarning: Insufficient balance to meet investment goal."
            advice += f"\nExpected Annual Return (AAPL): {self._avg_returns['stocks']*100:.1f}%"
            advice += f"\nVolatility (Risk): {self._volatility['stocks']*100:.1f}%"
            advice += f"\n*** Data Status: {self._data_status} ***"
            return advice
        except Exception as e:
            return f"Error generating investment advice: {str(e)}. Ensure valid inputs and API key."

    def optimize_allocation(self):
        """Optimize allocation using real-time returns and volatility."""
        try:
            balance = self._budget_manager.calculate_balance()
            if balance <= 0:
                return "No funds available to optimize."

            portfolio = InvestmentPortfolio(balance)
            current_alloc = portfolio._allocations
            risk_tolerance = portfolio._risk_tolerance

            stock_weight = self._avg_returns["stocks"] / self._volatility["stocks"]
            bond_weight = self._avg_returns["bonds"] / self._volatility["bonds"]
            cash_weight = self._avg_returns["cash"] / self._volatility["cash"]
            total_weight = stock_weight + bond_weight + cash_weight

            if risk_tolerance == "high":
                current_alloc["stocks"] = min(0.9, stock_weight / total_weight)
                current_alloc["bonds"] = max(0.0, bond_weight / total_weight)
                current_alloc["cash"] = max(0.1, 1 - current_alloc["stocks"] - current_alloc["bonds"])
            elif risk_tolerance == "low":
                current_alloc["stocks"] = max(0.3, stock_weight / total_weight * 0.5)
                current_alloc["bonds"] = min(0.5, bond_weight / total_weight * 1.5)
                current_alloc["cash"] = max(0.2, 1 - current_alloc["stocks"] - current_alloc["bonds"])
            else:  # moderate
                current_alloc["stocks"] = stock_weight / total_weight
                current_alloc["bonds"] = bond_weight / total_weight
                current_alloc["cash"] = 1 - current_alloc["stocks"] - current_alloc["bonds"]

            portfolio._allocations = current_alloc
            return portfolio.get_portfolio_summary()
        except Exception as e:
            return f"Error optimizing allocation: {str(e)}. Check data and inputs."

# Streamlit Dashboard and GUI
def run_budget_tracker():
    """Streamlit-based GUI for Budget Tracker."""
    st.title("Budget Tracker")

    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = User("Alice")
    if 'budget_mgr' not in st.session_state:
        st.session_state.budget_mgr = BudgetManager(st.session_state.user)
    if 'report_gen' not in st.session_state:
        st.session_state.report_gen = ReportGenerator(st.session_state.budget_mgr)
    if 'advisor' not in st.session_state:
        st.session_state.advisor = InvestmentAdvisor(st.session_state.budget_mgr)

    # Sidebar menu
    menu = ["Add Transaction", "Set Savings Goal", "Generate Report", "Create Pie Chart", "Suggest Investment", "Run ETL", "Ad-Hoc Query", "Forecast Returns"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Add Transaction
    if choice == "Add Transaction":
        st.header("Add Transaction")
        amount = st.number_input("Amount", min_value=0.0, step=0.01)
        category = st.text_input("Category", value="General")
        trans_type = st.selectbox("Type", ["income", "expense"])
        if st.button("Add"):
            t = Transaction(amount, category, trans_type)
            if st.session_state.budget_mgr.add_transaction(t):
                st.success("Transaction added!")
            else:
                st.error("Invalid transaction.")

    # Set Savings Goal
    elif choice == "Set Savings Goal":
        st.header("Set Savings Goal")
        goal = st.number_input("Goal Amount", min_value=0.0, step=0.01)
        if st.button("Set"):
            if st.session_state.user.update_savings_goal(goal):
                st.success("Goal set!")
            else:
                st.error("Invalid goal.")

    # Generate Report
    elif choice == "Generate Report":
        st.header("Budget Report")
        report = st.session_state.budget_mgr.generate_report()
        st.text(report)

    # Create Pie Chart
    elif choice == "Create Pie Chart":
        st.header("Expense Distribution")
        buf = st.session_state.report_gen.create_pie_chart()
        if isinstance(buf, io.BytesIO):
            st.image(buf)
        else:
            st.error(buf)

    # Suggest Investment
    elif choice == "Suggest Investment":
        st.header("Investment Suggestion")
        goal = st.number_input("Goal Amount", min_value=0.0, step=0.01)
        risk = st.selectbox("Risk Tolerance", ["low", "moderate", "high"])
        historical_returns = st.text_input("Historical Returns (comma-separated, e.g., 0.02,0.03,0.01)", value="0.02,0.03,0.01")
        if st.button("Suggest"):
            returns_list = [float(r) for r in historical_returns.split(',')]
            suggestion = st.session_state.advisor.suggest_investment(goal, risk, returns_list)
            st.text(suggestion)

    # Run ETL
    elif choice == "Run ETL":
        st.header("Run ETL from CSV")
        csv_path = st.text_input("CSV Path", value="sample_transactions.csv")
        if st.button("Run"):
            result = st.session_state.budget_mgr.db.etl_from_csv(csv_path)
            st.text(result)

    # Ad-Hoc Query
    elif choice == "Ad-Hoc Query":
        st.header("Ad-Hoc SQL Query")
        query = st.text_input("Enter SQL Query", "SELECT * FROM transactions")
        if st.button("Run Query"):
            result = st.session_state.budget_mgr.db.ad_hoc_query(query)
            st.dataframe(result)

    # Forecast Returns
    elif choice == "Forecast Returns":
        st.header("Forecast Returns")
        historical_returns = st.text_input("Historical Returns (comma-separated)", value="0.02,0.03,0.01")
        if st.button("Forecast"):
            returns_list = [float(r) for r in historical_returns.split(',')]
            pred = st.session_state.advisor._predictor.forecast_returns(returns_list)
            st.write(f"Predicted Next Return: {pred:.4f}")

# Run the app in Colab or GitHub
run_budget_tracker()
