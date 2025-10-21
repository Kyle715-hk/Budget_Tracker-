Budget Tracker - Streamlit Dashboard & GUI Version

Welcome to the Budget Tracker, a robust financial management tool designed to showcase my skills as a candidate for the Rev-celerator Graduate Programme 2026: Data Scientist and Analyst at Revolut. This Streamlit-based GUI version leverages Python, machine learning with Scikit-learn (linear regression) and Statsmodels (ARIMA forecasting), and real-time data integration with Alpha Vantage API to provide an interactive experience for tracking finances, analyzing spending, and suggesting investments.

Project Overview

Built with Streamlit, Pandas, NumPy, Scikit-learn, Statsmodels, and the Alpha Vantage API, this application offers:





Transaction Tracking: Add income and expenses with category and date logging.



Budget Management: Set savings goals and monitor progress.



Data Visualization: Generate pie charts for expense distribution.



Investment Insights: ML-driven portfolio suggestions using Linear Regression and ARIMA forecasting.



Data Handling: ETL from CSV and ad-hoc SQL queries on a SQLite database.

This project highlights my expertise in data science, including data preprocessing, machine learning, and deployment—key skills for the Data Scientist and Analyst role at Revolut.

Getting Started

Prerequisites





Python 3.8+



Install dependencies:

pip install streamlit pandas numpy scikit-learn statsmodels python-dotenv matplotlib alpha_vantage

Setup





Clone the repository:

git clone https://github.com/Kyle715-hk/Budget_Tracker-.git
cd Budget_Tracker-



Configure your Alpha Vantage API key:





Create a .env file with ALPHA_VANTAGE_API_KEY=your_key_here (get a free key from Alpha Vantage).



Alternatively, use Streamlit Secrets for deployment.



Run the app:

streamlit run app.py



Access the app at http://localhost:8501.

Live Demo

Explore the app live: [Budget Tracker Streamlit App.](https://budget-tracker-kyle.streamlit.app/)

Features





Add Transaction: Log income or expenses (ensure amounts are positive; expenses cannot exceed income).



Set Savings Goal: Define and track savings targets.



Generate Report: View a detailed financial summary.



Create Pie Chart: Visualize expense categories.



Suggest Investment: Get ML-based portfolio advice with real-time AAPL data.



Run ETL: Import transactions from a CSV file.



Ad-Hoc Query: Execute custom SQL queries on transaction data.



Forecast Returns: Predict future returns using ARIMA.

Troubleshooting





Invalid Transaction: Occurs if the amount is zero/negative or if an expense exceeds income. Add income first or adjust the amount.



sqlite3.ProgrammingError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if on Streamlit Cloud, click 'Manage app' in the lower right). Switch to the Command-Line Menu Version for a stable alternative.



API Issues: Ensure your Alpha Vantage API key is valid (free tier is rate-limited).

Why This Matters for Rev-celerator 2026

This project demonstrates:





Data Science Skills: ML models (Linear Regression, ARIMA) for investment forecasting.



Data Engineering: ETL processes and SQL querying.



Deployment: Streamlit for interactive web apps.



Problem-Solving: Handling real-world errors like database issues.

As a candidate, I’m eager to apply these skills at Revolut to drive data-driven financial innovations. Contact me at wangtikchan715@gmail.com or LinkedIn for further discussion.

License

MIT License - See the LICENSE file for details.
