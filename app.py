import streamlit as st
import google.generativeai as genai
import os
import requests
import json
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
gemini_api_key = st.secrets["gemini_api_key"]
serper_api_key = st.secrets["serper_api_key"]
genai.configure(api_key=gemini_api_key)
st.set_page_config(page_title="Fin-AI-lytics", layout="wide")
st.title("Fin-AI-lytics")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "financial_context" not in st.session_state:
    st.session_state.financial_context = {
        "goals": [],
        "budget": {},
        "investments": [],
        "savings": {},
        "expenses": [],
        "portfolio": []
    }
if "market_data" not in st.session_state:
    st.session_state.market_data = {
        "latest_analysis": None,
        "sentiment": None,
        "last_updated": None
    }
if "budget_categories" not in st.session_state:
    st.session_state.budget_categories = ["Housing", "Food", "Transportation", "Utilities", "Entertainment", "Healthcare", "Shopping", "Other"]
def get_market_news(query, num_results=5):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": num_results
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()
    except Exception as e:
        st.error(f"Error fetching market news: {e}")
        return None
def detect_fraud_in_transactions(transactions):
    fraud_indicators = []
    if len(transactions) > 0:
        sorted_transactions = sorted(transactions, key=lambda x: x.get('date', ''))
        avg_amount = sum(t.get('amount', 0) for t in transactions) / len(transactions)
        threshold = avg_amount * 3 
        
        for transaction in transactions:
            fraud_score = 0
            reasons = []
            if transaction.get('amount', 0) > threshold:
                fraud_score += 30
                reasons.append("Unusually large transaction amount")
            if transaction.get('location_usual', True) == False:
                fraud_score += 25
                reasons.append("Unusual transaction location")
            if transaction.get('category_usual', True) == False:
                fraud_score += 20
                reasons.append("Unusual merchant category")
            if transaction.get('time_usual', True) == False:
                fraud_score += 15
                reasons.append("Unusual transaction time")
            if transaction.get('rapid_succession', False):
                fraud_score += 25
                reasons.append("Multiple transactions in rapid succession")
            if fraud_score >= 40:
                transaction['fraud_score'] = fraud_score
                transaction['fraud_reasons'] = reasons
                fraud_indicators.append(transaction)
    
    return fraud_indicators
if "transactions" not in st.session_state:
    st.session_state.transactions = []

def analyze_market_sentiment(news_data):
    if not news_data or 'organic' not in news_data:
        return "Unable to analyze market sentiment due to insufficient data."
    news_articles = []
    for article in news_data.get('organic', []):
        news_articles.append({
            'title': article.get('title', ''),
            'snippet': article.get('snippet', ''),
            'date': article.get('date', '')
        })
    prompt = f"""
    Consider you are the best financial scientist in the entire world.
    Make sure you directly stick to the topic and do not beat around the bush due to poor news quality or due to being news being off topic.
    Analyze the following financial news articles for market sentiment:
    {json.dumps(news_articles, indent=2)}
    Based on these articles:
    1. Provide a detailed summary of about 2500 words of current market conditions
    2. Analyze the overall market sentiment (bullish, bearish, or neutral)
    3. Identify key trends or patterns mentioned
    4. Give a market outlook prediction (positive, negative, or uncertain)
    Format your response as a JSON with these keys: summary, sentiment, trends, outlook, confidence_score (0-100)
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    try:
        analysis_text = response.text
        json_start = analysis_text.find('{')
        json_end = analysis_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            analysis_json = json.loads(analysis_text[json_start:json_end])
            return analysis_json
        else:
            return {"error": "Could not parse JSON from response", "raw_response": analysis_text}
    except Exception as e:
        return {"error": str(e), "raw_response": response.text}
def get_gemini_response(messages, financial_context, market_data):
    system_prompt = f"""
    You are a Smart Financial Assistant with the following capabilities:
    - Real-time query handling for financial questions
    - Context-aware responses based on user goals
    User's financial context:
    Goals: {financial_context['goals']}
    Budget: {financial_context['budget']}
    Investments: {financial_context['investments']}
    Savings: {financial_context['savings']}
    Expenses: {financial_context['expenses']}
    Portfolio: {financial_context['portfolio']}
    Recent market analysis:
    {market_data.get('latest_analysis', 'No recent market analysis available')}
    Sentiment: {market_data.get('sentiment', 'Unknown')}
    Last updated: {market_data.get('last_updated', 'Never')}
    Respond in a helpful, professional manner. Provide specific, actionable financial advice based on the user's goals and context.
    """
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_prompt
    )
    conversation = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        conversation.append({"role": role, "parts": [msg["content"]]})
    response = model.generate_content(conversation)
    return response.text
def run_automatic_market_analysis():
    analysis_targets = [
        "current stock market trends and outlook",
        "S&P 500 latest performance and forecast",
        "NASDAQ market sentiment this week",
        "global financial market conditions",
        "market volatility indicators current status"
    ]
    import random
    target = random.choice(analysis_targets)
    news_data = get_market_news(target)
    if news_data:
        analysis = analyze_market_sentiment(news_data)
        st.session_state.market_data = {
            "latest_analysis": analysis,
            "sentiment": analysis.get("sentiment", "Unknown"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "news_data": news_data,
            "target": target
        }
        return True
    return False
def predict_future_prices(ticker, models, scalers, days=30, window_size=30):
    try:
        data = get_stock_data_past_year(ticker)
        scaler = scalers[ticker]
        scaled_data = scaler.transform(data.values.reshape(-1, 1)).flatten()
        last_window = scaled_data[-window_size:]
        model_info = models[ticker]
        model_type = model_info['type']
        predictions = []
        if model_type == 'ensemble':
            xgb_preds, lstm_preds, lgbm_preds = [], [], []
            current_window_xgb = last_window.copy()
            current_window_lstm = last_window.copy()
            current_window_lgbm = last_window.copy()
            for _ in range(days):
                xgb_pred = model_info['model']['xgb'].predict(current_window_xgb.reshape(1, -1))[0]
                lstm_pred = model_info['model']['lstm'].predict(current_window_lstm.reshape(1, window_size, 1))[0][0]
                lgbm_pred = model_info['model']['lgbm'].predict(current_window_lgbm.reshape(1, -1))[0]
                xgb_preds.append(xgb_pred)
                lstm_preds.append(lstm_pred)
                lgbm_preds.append(lgbm_pred)
                current_window_xgb = np.append(current_window_xgb[1:], xgb_pred)
                current_window_lstm = np.append(current_window_lstm[1:], lstm_pred)
                current_window_lgbm = np.append(current_window_lgbm[1:], lgbm_pred)
            predictions = [(xgb_preds[i] + lstm_preds[i] + lgbm_preds[i]) / 3 for i in range(days)]
        else:
            current_window = last_window.copy()
            model = model_info['model']
            for _ in range(days):
                if model_type == 'lstm':
                    prediction = model.predict(current_window.reshape(1, window_size, 1))[0][0]
                else:
                    prediction = model.predict(current_window.reshape(1, -1))[0]
                predictions.append(prediction)
                current_window = np.append(current_window[1:], prediction)
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None
def get_stock_data_past_year(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock['Close']
def create_features(data, window_size=30):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
    return np.array(X)
def load_prediction_models():
    models = {}
    scalers = {}
    try:
        with open('xgb_nvidia_model.pkl', 'rb') as f:
            models['NVDA'] = {'model': pickle.load(f), 'type': 'xgb'}
        models['TSLA'] = {'model': load_model('lstm_tesla_model.h5'), 'type': 'lstm'}
        with open('lgbm_google_model.pkl', 'rb') as f:
            models['GOOGL'] = {'model': pickle.load(f), 'type': 'lgbm'}
        with open('xgb_apple_model.pkl', 'rb') as f:
            xgb_apple = pickle.load(f)
        lstm_apple = load_model('lstm_apple_model.h5')
        with open('lgbm_apple_model.pkl', 'rb') as f:
            lgbm_apple = pickle.load(f)
        models['AAPL'] = {
            'model': {'xgb': xgb_apple, 'lstm': lstm_apple, 'lgbm': lgbm_apple},
            'type': 'ensemble'
        }
        for ticker in ['NVDA', 'TSLA', 'GOOGL', 'AAPL']:
            data = get_stock_data_past_year(ticker)
            scaler = MinMaxScaler()
            scaler.fit(data.values.reshape(-1, 1))
            scalers[ticker] = scaler
        return models, scalers
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None
def predict_next_month(ticker, models, scalers, window_size=30):
    try:
        data = get_stock_data_past_year(ticker)
        scaler = scalers[ticker]
        scaled_data = scaler.transform(data.values.reshape(-1, 1)).flatten()
        last_window = scaled_data[-window_size:]
        model_info = models[ticker]
        model_type = model_info['type']
        predictions = []
        if model_type == 'ensemble':
            xgb_preds, lstm_preds, lgbm_preds = [], [], []
            current_window_xgb, current_window_lstm, current_window_lgbm = last_window.copy(), last_window.copy(), last_window.copy()
            for _ in range(30):
                xgb_input = current_window_xgb.reshape(1, -1)
                xgb_pred = model_info['model']['xgb'].predict(xgb_input)[0]
                xgb_preds.append(xgb_pred)
                current_window_xgb = np.append(current_window_xgb[1:], xgb_pred)
                lstm_input = current_window_lstm.reshape(1, window_size, 1)
                lstm_pred = model_info['model']['lstm'].predict(lstm_input)[0][0]
                lstm_preds.append(lstm_pred)
                current_window_lstm = np.append(current_window_lstm[1:], lstm_pred)
                lgbm_input = current_window_lgbm.reshape(1, -1)
                lgbm_pred = model_info['model']['lgbm'].predict(lgbm_input)[0]
                lgbm_preds.append(lgbm_pred)
                current_window_lgbm = np.append(current_window_lgbm[1:], lgbm_pred)
            for i in range(30):
                predictions.append((xgb_preds[i] + lstm_preds[i] + lgbm_preds[i]) / 3)
        else:
            current_window = last_window.copy()
            model = model_info['model']
            for _ in range(30):
                if model_type == 'lstm':
                    model_input = current_window.reshape(1, window_size, 1)
                    prediction = model.predict(model_input)[0][0]
                else:
                    model_input = current_window.reshape(1, -1)
                    prediction = model.predict(model_input)[0]
                predictions.append(prediction)
                current_window = np.append(current_window[1:], prediction)
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None
def optimize_portfolio(portfolio, predictions):
    if not portfolio:
        return {"buy": [], "hold": [], "sell": []}
    optimization_results = {"buy": [], "hold": [], "sell": []}
    for stock in portfolio:
        ticker = stock["ticker"]
        current_price = stock["current_price"]
        if ticker in predictions:
            predicted_price = predictions[ticker][-1]
            expected_return = (predicted_price - current_price) / current_price * 100
            if expected_return > 5:
                optimization_results["buy"].append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "expected_return": expected_return,
                    "recommendation": "Buy more"
                })
            elif expected_return < -2:
                optimization_results["sell"].append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "expected_return": expected_return,
                    "recommendation": "Consider selling"
                })
            else:
                optimization_results["hold"].append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "expected_return": expected_return,
                    "recommendation": "Hold"
                })
    return optimization_results
def calculate_progress_to_goals(goals, savings, expenses):
    if not goals:
        return []
    progress_data = []
    for goal in goals:
        target_amount = goal["target_amount"]
        target_date = datetime.strptime(goal["target_date"], "%Y-%m-%d")
        days_remaining = (target_date - datetime.now()).days
        if days_remaining < 0:
            progress_percentage = 0
            status = "Expired"
        else:
            total_savings = sum(item["amount"] for item in savings.get("accounts", []))
            monthly_expenses = sum(expense["amount"] for expense in expenses)
            progress_percentage = min(100, (total_savings / target_amount) * 100)
            if progress_percentage >= 100:
                status = "Achieved"
            elif days_remaining <= 30 and progress_percentage < 90:
                status = "At Risk"
            else:
                status = "On Track"
        progress_data.append({
            "goal": goal["goal"],
            "target_amount": target_amount,
            "target_date": goal["target_date"],
            "progress_percentage": progress_percentage,
            "status": status,
            "days_remaining": max(0, days_remaining)
        })
    return progress_data
with st.sidebar:
    st.header("Navigation")
    selected_tab = st.radio(
        "Select a tab:",
        ["Chat Assistant", "Personal Finance", "Market Analysis", "Investments", "Smart Dashboard", "Fraud Detection"],
        key="tab_selector"
    )

if selected_tab == "Chat Assistant":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask your financial assistant..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_gemini_response(
                    st.session_state.messages, 
                    st.session_state.financial_context,
                    st.session_state.market_data
                )
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
elif selected_tab == "Personal Finance":
    st.header("Personal Finance Manager")
    pf_tab1, pf_tab2, pf_tab3 = st.tabs(["Financial Goals", "Expense Tracking", "Budget Planning"])
    with pf_tab1:
        st.subheader("Financial Goals")
        col1, col2 = st.columns([2, 1])
        with col1:
            new_goal = st.text_input("Add a new financial goal")
            target_amount = st.number_input("Target Amount ($)", min_value=0, step=1000)
            target_date = st.date_input("Target Date")
            if st.button("Add Goal"):
                if new_goal:
                    goal_details = {
                        "goal": new_goal,
                        "target_amount": target_amount,
                        "target_date": target_date.strftime("%Y-%m-%d"),
                        "created_at": datetime.now().strftime("%Y-%m-%d"),
                        "progress": 0
                    }
                    st.session_state.financial_context["goals"].append(goal_details)
                    st.success(f"Goal added: {new_goal}")
        with col2:
            if not st.session_state.financial_context["savings"]:
                st.session_state.financial_context["savings"] = {"accounts": []}
            savings_amount = st.number_input("Current Savings Amount ($)", min_value=0, step=100)
            savings_name = st.text_input("Savings Account Name", value="General Savings")
            if st.button("Update Savings"):
                existing_account = False
                for account in st.session_state.financial_context["savings"].get("accounts", []):
                    if account["name"] == savings_name:
                        account["amount"] = savings_amount
                        existing_account = True
                        break
                if not existing_account:
                    if "accounts" not in st.session_state.financial_context["savings"]:
                        st.session_state.financial_context["savings"]["accounts"] = []
                    st.session_state.financial_context["savings"]["accounts"].append({
                        "name": savings_name,
                        "amount": savings_amount,
                        "updated_at": datetime.now().strftime("%Y-%m-%d")
                    })
                st.success(f"Savings updated: {savings_name} - ${savings_amount}")
        st.subheader("Your Goals Progress")
        if st.session_state.financial_context["goals"]:
            progress_data = calculate_progress_to_goals(
                st.session_state.financial_context["goals"],
                st.session_state.financial_context["savings"],
                st.session_state.financial_context["expenses"]
            )
            for idx, progress in enumerate(progress_data):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"**{progress['goal']}**")
                    st.progress(progress['progress_percentage']/100)
                with col2:
                    st.write(f"${progress['target_amount']}")
                with col3:
                    st.write(f"{progress['days_remaining']} days left")
                with col4:
                    if progress['status'] == "On Track":
                        st.success(progress['status'])
                    elif progress['status'] == "At Risk":
                        st.warning(progress['status'])
                    elif progress['status'] == "Achieved":
                        st.info(progress['status'])
                    else:
                        st.error(progress['status'])
                if st.button(f"Remove Goal {idx+1}"):
                    st.session_state.financial_context["goals"].pop(idx)
                    st.rerun()
        else:
            st.info("No financial goals added yet. Add your first goal above.")
    with pf_tab2:
        st.subheader("Expense Tracker")
        col1, col2 = st.columns([1, 1])
        with col1:
            expense_amount = st.number_input("Expense Amount ($)", min_value=0, step=10)
            expense_category = st.selectbox("Category", st.session_state.budget_categories)
            expense_date = st.date_input("Date of Expense", value=datetime.now())
            expense_note = st.text_input("Note (optional)")
            if st.button("Add Expense"):
                expense = {
                    "amount": expense_amount,
                    "category": expense_category,
                    "date": expense_date.strftime("%Y-%m-%d"),
                    "note": expense_note,
                    "id": len(st.session_state.financial_context["expenses"]) + 1
                }
                st.session_state.financial_context["expenses"].append(expense)
                st.success(f"Expense added: ${expense_amount} for {expense_category}")
        with col2:
            new_category = st.text_input("Add Custom Category")
            if st.button("Add Category") and new_category and new_category not in st.session_state.budget_categories:
                st.session_state.budget_categories.append(new_category)
                st.success(f"Category added: {new_category}")
        st.subheader("Recent Expenses")
        if st.session_state.financial_context["expenses"]:
            expense_data = pd.DataFrame(st.session_state.financial_context["expenses"])
            expense_data = expense_data.sort_values(by="date", ascending=False)
            st.dataframe(expense_data, use_container_width=True)
            if len(expense_data) > 0:
                st.subheader("Expense Analysis")
                category_totals = expense_data.groupby("category")["amount"].sum().reset_index()
                fig = plt.figure(figsize=(10, 6))
                plt.pie(category_totals["amount"], labels=category_totals["category"], autopct="%1.1f%%")
                plt.title("Expense Distribution by Category")
                st.pyplot(fig)
        else:
            st.info("No expenses recorded yet. Add your first expense above.")
        if st.button("Clear All Expenses"):
            st.session_state.financial_context["expenses"] = []
            st.success("All expenses cleared!")
            st.rerun()
    with pf_tab3:
        st.subheader("Budget Planning")
        col1, col2 = st.columns([1, 1])
        with col1:
            budget_category = st.selectbox("Budget Category", st.session_state.budget_categories, key="budget_cat")
            budget_amount = st.number_input("Monthly Budget ($)", min_value=0, step=100)
            if st.button("Set Budget"):
                if "budgets" not in st.session_state.financial_context["budget"]:
                    st.session_state.financial_context["budget"]["budgets"] = {}
                st.session_state.financial_context["budget"]["budgets"][budget_category] = budget_amount
                st.success(f"Budget set: ${budget_amount} for {budget_category}")
        with col2:
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, step=100)
            if st.button("Set Income"):
                st.session_state.financial_context["budget"]["income"] = monthly_income
                st.success(f"Monthly income set: ${monthly_income}")
        st.subheader("Your Budget vs. Actual Spending")
        if "budgets" in st.session_state.financial_context["budget"]:
            budgets = st.session_state.financial_context["budget"]["budgets"]
            if budgets and st.session_state.financial_context["expenses"]:
                expense_df = pd.DataFrame(st.session_state.financial_context["expenses"])
                current_month = datetime.now().strftime("%Y-%m")
                expense_df["month"] = expense_df["date"].apply(lambda x: x[:7])
                monthly_expenses = expense_df[expense_df["month"] == current_month]
                expense_by_category = monthly_expenses.groupby("category")["amount"].sum().to_dict()
                budget_comparison = []
                for category, budget in budgets.items():
                    actual = expense_by_category.get(category, 0)
                    remaining = budget - actual
                    percentage = min(100, (actual / budget) * 100) if budget > 0 else 0
                    budget_comparison.append({
                        "category": category,
                        "budget": budget,
                        "actual": actual,
                        "remaining": remaining,
                        "percentage": percentage
                    })
                budget_df = pd.DataFrame(budget_comparison)
                st.dataframe(budget_df, use_container_width=True)
                for idx, row in budget_df.iterrows():
                    st.write(f"**{row['category']}**: ${row['actual']} of ${row['budget']}")
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        progress_color = "normal"
                        if row['percentage'] > 90:
                            progress_color = "red"
                        elif row['percentage'] > 75:
                            progress_color = "orange"
                        st.progress(row['percentage']/100)
                    with col2:
                        st.write(f"${row['remaining']} left")
                if "income" in st.session_state.financial_context["budget"]:
                    income = st.session_state.financial_context["budget"]["income"]
                    total_expenses = expense_by_category.values()
                    total_monthly_expense = sum(total_expenses) if total_expenses else 0
                    savings_potential = income - total_monthly_expense
                    st.subheader("Monthly Overview")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Income", f"${income}")
                    with col2:
                        st.metric("Expenses", f"${total_monthly_expense}")
                    with col3:
                        st.metric("Potential Savings", f"${savings_potential}")
            else:
                st.info("Set budgets and record expenses to see your budget analysis.")
        else:
            st.info("No budgets set yet. Create your first budget above.")
elif selected_tab == "Market Analysis":
    st.header("AI-powered Market Analysis")
    col1, col2 = st.columns([4, 1])
    with col2:
        refresh = st.button("Refresh Analysis")
    if refresh or st.session_state.market_data["latest_analysis"] is None:
        with st.spinner("Analyzing market conditions..."):
            success = run_automatic_market_analysis()
            if success:
                st.success("Market analysis updated!")
            else:
                st.error("Failed to update market analysis. Please try again later.")
    if st.session_state.market_data.get("latest_analysis"):
        analysis = st.session_state.market_data["latest_analysis"]
        st.caption(f"Analysis based on: {st.session_state.market_data.get('target', 'market conditions')}")
        sentiment = analysis.get("sentiment", "").lower()
        if "bull" in sentiment:
            sentiment_color = "green"
            sentiment_emoji = "📈"
        elif "bear" in sentiment:
            sentiment_color = "red"
            sentiment_emoji = "📉"
        else:
            sentiment_color = "gray"
            sentiment_emoji = "⚖️"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <h3 style="margin-right: 10px;">Market Sentiment:</h3>
            <div style="background-color: {sentiment_color}; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block;">
                {sentiment_emoji} {analysis.get("sentiment", "Unknown").upper()}
            </div>
            <div style="margin-left: 20px; color: gray; font-size: 0.8em;">
                Confidence: {analysis.get("confidence_score", "N/A")}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Market Summary")
        st.write(analysis.get("summary", "No summary available"))
        st.subheader("Key Trends")
        trends = analysis.get("trends", [])
        if isinstance(trends, list) and trends:
            formatted_trends = "\n".join([f"{i+1}. {trend}" for i, trend in enumerate(trends)])
            st.write(formatted_trends)
        else:
            st.write("No trends identified")
        st.subheader("Market Outlook")
        outlook = analysis.get("outlook", "").lower()
        if "positive" in outlook:
            st.success(analysis.get("outlook", "Uncertain"))
        elif "negative" in outlook:
            st.error(analysis.get("outlook", "Uncertain"))
        else:
            st.info(analysis.get("outlook", "Uncertain"))
        news_data = st.session_state.market_data.get("news_data", {})
        if "organic" in news_data:
            with st.expander("View News Sources"):
                for i, article in enumerate(news_data["organic"][:5], 1):
                    st.markdown(f"**{i}. {article.get('title', 'No title')}**")
                    st.markdown(f"{article.get('snippet', 'No snippet')}")
                    if article.get('link'):
                        st.markdown(f"[Read more]({article.get('link')})")
                    st.divider()
        st.caption(f"Last updated: {st.session_state.market_data['last_updated']}")
    else:
        st.info("No market analysis available. Click 'Refresh Analysis' to generate one.")
elif selected_tab == "Investments":
    st.header("Investment Manager")
    inv_tab1, inv_tab2, inv_tab3 = st.tabs(["Stock Predictions", "Portfolio Tracker", "Portfolio Optimization"])
    
    with inv_tab1:
        st.subheader("Stock Price Predictions")
        if "models" not in st.session_state or "scalers" not in st.session_state:
            with st.spinner("Loading prediction models..."):
                models, scalers = load_prediction_models()
                if models and scalers:
                    st.session_state.models = models
                    st.session_state.scalers = scalers
                else:
                    st.error("Failed to load models. Please make sure model files are in the correct location.")
        
        stock_options = {
            'NVDA': 'NVIDIA',
            'TSLA': 'Tesla',
            'GOOGL': 'Google',
            'AAPL': 'Apple'
        }
        
        selected_stock = st.selectbox(
            "Select a stock to predict:",
            options=list(stock_options.keys()),
            format_func=lambda x: stock_options[x],
            key="prediction_stock"
        )
        
        if "models" in st.session_state and "scalers" in st.session_state:
            with st.spinner(f"Fetching historical data for {selected_stock}..."):
                historical_data = get_stock_data_past_year(selected_stock)
                predictions = predict_next_month(
                    selected_stock, 
                    st.session_state.models, 
                    st.session_state.scalers
                )
                
                if predictions is not None:
                    last_date = historical_data.index[-1]
                    prediction_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=30,
                        freq='D'
                    )
                    prediction_dates = prediction_dates[prediction_dates.dayofweek < 5]
                    
                    one_year_ago = pd.Timestamp.today() - pd.DateOffset(years=1)
                    historical_data = historical_data[historical_data.index >= one_year_ago]
                    
                    hist_tab, pred_tab = st.tabs(["Historical (1 Year)", "Prediction (1 Month)"])
                    
                    with hist_tab:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(historical_data.index, historical_data.values, color='blue', linewidth=2, label='Historical')
                        ax.scatter(historical_data.index[-1], historical_data.values[-1], color='green', s=100, label='Last Close')
                        ax.set_title(f"{selected_stock} Historical Stock Price (Last Year)")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price ($)")
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig)
                        
                        hist_min = float(historical_data.min())
                        hist_max = float(historical_data.max())
                        hist_start = float(historical_data.iloc[0])
                        hist_end = float(historical_data.iloc[-1])
                        hist_change_pct = (hist_end - hist_start) / hist_start * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Starting Price (1yr ago)", f"${hist_start:.2f}")
                        with col2:
                            st.metric("Current Price", f"${float(historical_data.values[-1]):.2f}", 
                                    f"{hist_change_pct:.2f}%")
                    
                    with pred_tab:
                        last_known_date = historical_data.index[-1]
                        predicted_final_date = prediction_dates[-1]
                        last_known_price = float(historical_data.values[-1])
                        predicted_final_price = float(predictions[-1])
                        
                        fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
                        last_30_days = historical_data.iloc[-30:]
                        ax_pred.plot(last_30_days.index, last_30_days.values, color='blue', linewidth=2, label='Historical')
                        ax_pred.plot(
                            [last_known_date, predicted_final_date], 
                            [last_known_price, predicted_final_price],
                            color='red', linewidth=2, linestyle='dashed', label='Prediction (Straight Line)'
                        )
                        ax_pred.scatter(last_known_date, last_known_price, color='green', s=100, label='Last Close')
                        ax_pred.set_title(f"{selected_stock} Stock Price Prediction (Next Month)")
                        ax_pred.set_xlabel("Date")
                        ax_pred.set_ylabel("Price ($)")
                        ax_pred.legend()
                        ax_pred.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig_pred)
                        
                        pred_change_pct = (predicted_final_price - last_known_price) / last_known_price * 100
                        pred_max = float(np.max(predictions))
                        pred_min = float(np.min(predictions))
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${last_known_price:.2f}")
                        with col2:
                            st.metric("Predicted (30 days)", f"${predicted_final_price:.2f}", f"{pred_change_pct:.2f}%")
                        with col3:
                            st.metric("Predicted Range", f"${pred_min:.2f} - ${pred_max:.2f}")
                    
                    st.info(
                        "**Disclaimer**: These predictions are for informational purposes only. "
                        "Stock market investments involve risk. "
                    )
                else:
                    st.error("Failed to generate predictions for the selected stock.")
        else:
            st.warning("Models are not loaded. Please try refreshing the page.")
    
    with inv_tab2:
        st.subheader("Portfolio Tracker")
        
        if not st.session_state.financial_context["portfolio"]:
            st.session_state.financial_context["portfolio"] = []
        
        col1, col2, col3 = st.columns(3)
        with col1:
            stock_ticker = st.text_input("Stock Ticker Symbol", key="portfolio_ticker").upper()
        with col2:
            shares = st.number_input("Number of Shares", min_value=0.01, step=0.01, key="portfolio_shares")
        with col3:
            purchase_price = st.number_input("Purchase Price per Share ($)", min_value=0.01, step=0.01, key="portfolio_price")
        
        purchase_date = st.date_input("Purchase Date", key="portfolio_date")
        notes = st.text_input("Notes (optional)", key="portfolio_notes")
        
        if st.button("Add to Portfolio"):
            if stock_ticker and shares > 0 and purchase_price > 0:
                try:
                    stock = yf.Ticker(stock_ticker)
                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                    
                    current_value = current_price * shares
                    purchase_value = purchase_price * shares
                    profit_loss = current_value - purchase_value
                    profit_loss_percent = (profit_loss / purchase_value) * 100
                    
                    portfolio_entry = {
                        "ticker": stock_ticker,
                        "shares": shares,
                        "purchase_price": purchase_price,
                        "purchase_date": purchase_date.strftime("%Y-%m-%d"),
                        "purchase_value": purchase_value,
                        "current_price": current_price,
                        "current_value": current_value,
                        "profit_loss": profit_loss,
                        "profit_loss_percent": profit_loss_percent,
                        "notes": notes,
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.session_state.financial_context["portfolio"].append(portfolio_entry)
                    st.success(f"Added {shares} shares of {stock_ticker} to portfolio!")
                except Exception as e:
                    st.error(f"Error: {e}. Please check the ticker symbol and try again.")
            else:
                st.warning("Please fill in all required fields.")
        
        if st.session_state.financial_context["portfolio"]:
            portfolio = st.session_state.financial_context["portfolio"]
            
            if st.button("Refresh Portfolio Prices"):
                with st.spinner("Updating portfolio prices..."):
                    for i, stock in enumerate(portfolio):
                        try:
                            ticker_data = yf.Ticker(stock["ticker"])
                            current_price = ticker_data.history(period="1d")['Close'].iloc[-1]
                            
                            portfolio[i]["current_price"] = current_price
                            portfolio[i]["current_value"] = current_price * stock["shares"]
                            portfolio[i]["profit_loss"] = portfolio[i]["current_value"] - stock["purchase_value"]
                            portfolio[i]["profit_loss_percent"] = (portfolio[i]["profit_loss"] / stock["purchase_value"]) * 100
                            portfolio[i]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            st.warning(f"Could not update {stock['ticker']}. Skipping.")
                    st.success("Portfolio prices updated!")
            
            total_investment = sum(stock["purchase_value"] for stock in portfolio)
            total_current_value = sum(stock["current_value"] for stock in portfolio)
            total_profit_loss = total_current_value - total_investment
            total_profit_loss_percent = (total_profit_loss / total_investment) * 100 if total_investment > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Investment", f"${total_investment:.2f}")
            with col2:
                st.metric("Current Value", f"${total_current_value:.2f}")
            with col3:
                st.metric("Profit/Loss", f"${total_profit_loss:.2f}", f"{total_profit_loss_percent:.2f}%")
            
            portfolio_df = pd.DataFrame(portfolio)
            portfolio_df = portfolio_df[["ticker", "shares", "purchase_price", "current_price", 
                                        "purchase_value", "current_value", "profit_loss", "profit_loss_percent"]]
            
            formatted_df = portfolio_df.copy()
            formatted_df["purchase_price"] = formatted_df["purchase_price"].map("${:.2f}".format)
            formatted_df["current_price"] = formatted_df["current_price"].map("${:.2f}".format)
            formatted_df["purchase_value"] = formatted_df["purchase_value"].map("${:.2f}".format)
            formatted_df["current_value"] = formatted_df["current_value"].map("${:.2f}".format)
            formatted_df["profit_loss"] = formatted_df["profit_loss"].map("${:.2f}".format)
            formatted_df["profit_loss_percent"] = formatted_df["profit_loss_percent"].map("{:.2f}%".format)
            
            formatted_df.columns = ["Ticker", "Shares", "Buy Price", "Current Price", 
                                "Cost Basis", "Current Value", "Profit/Loss", "Return %"]
            
            st.dataframe(formatted_df, use_container_width=True)
            
            if len(portfolio) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=[stock["ticker"] for stock in portfolio],
                    values=[stock["current_value"] for stock in portfolio],
                    hole=.4,
                    textinfo='label+percent'
                )])
                fig.update_layout(title="Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)
                
                performance_data = [
                    go.Bar(
                        name="Cost Basis",
                        x=[stock["ticker"] for stock in portfolio],
                        y=[stock["purchase_value"] for stock in portfolio],
                        marker_color='blue'
                    ),
                    go.Bar(
                        name="Current Value",
                        x=[stock["ticker"] for stock in portfolio],
                        y=[stock["current_value"] for stock in portfolio],
                        marker_color='green'
                    )
                ]
                
                performance_layout = go.Layout(
                    title="Investment Performance by Stock",
                    barmode='group',
                    xaxis_title="Stock",
                    yaxis_title="Value ($)"
                )
                
                fig = go.Figure(data=performance_data, layout=performance_layout)
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Remove Selected Stock"):
                st.info("Select the stock to remove:")
                for i, stock in enumerate(portfolio):
                    if st.checkbox(f"{stock['ticker']} - {stock['shares']} shares", key=f"delete_{i}"):
                        st.session_state.financial_context["portfolio"].pop(i)
                        st.success(f"Removed {stock['ticker']} from portfolio!")
                        st.rerun()
        else:
            st.info("Your portfolio is empty. Add your first stock above!")
    
    with inv_tab3:
        st.subheader("Portfolio Optimization")
        
        if not st.session_state.financial_context["portfolio"]:
            st.info("Please add stocks to your portfolio first in the 'Portfolio Tracker' tab.")
        else:
            st.write("This tool analyzes your current portfolio and makes recommendations based on predicted price movements.")
            
            if st.button("Run Portfolio Analysis"):
                with st.spinner("Analyzing your portfolio..."):
                    portfolio = st.session_state.financial_context["portfolio"]
                    
                    predictions = {}
                    for stock in portfolio:
                        ticker = stock["ticker"]
                        if ticker in ["AAPL", "TSLA", "GOOGL", "NVDA"] and "models" in st.session_state:
                            pred = predict_next_month(ticker, st.session_state.models, st.session_state.scalers)
                            if pred is not None:
                                predictions[ticker] = pred
                    
                    recommendations = optimize_portfolio(portfolio, predictions)
                    
                    if recommendations["buy"]:
                        st.subheader("✅ Recommended Buys")
                        buy_df = pd.DataFrame(recommendations["buy"])
                        buy_df["expected_return"] = buy_df["expected_return"].map("{:.2f}%".format)
                        buy_df["current_price"] = buy_df["current_price"].map("${:.2f}".format)
                        buy_df["predicted_price"] = buy_df["predicted_price"].map("${:.2f}".format)
                        st.dataframe(buy_df[["ticker", "current_price", "predicted_price", "expected_return", "recommendation"]], use_container_width=True)
                    
                    if recommendations["hold"]:
                        st.subheader("⏸️ Recommended Holds")
                        hold_df = pd.DataFrame(recommendations["hold"])
                        hold_df["expected_return"] = hold_df["expected_return"].map("{:.2f}%".format)
                        hold_df["current_price"] = hold_df["current_price"].map("${:.2f}".format)
                        hold_df["predicted_price"] = hold_df["predicted_price"].map("${:.2f}".format)
                        st.dataframe(hold_df[["ticker", "current_price", "predicted_price", "expected_return", "recommendation"]], use_container_width=True)
                    
                    if recommendations["sell"]:
                        st.subheader("⚠️ Consider Selling")
                        sell_df = pd.DataFrame(recommendations["sell"])
                        sell_df["expected_return"] = sell_df["expected_return"].map("{:.2f}%".format)
                        sell_df["current_price"] = sell_df["current_price"].map("${:.2f}".format)
                        sell_df["predicted_price"] = sell_df["predicted_price"].map("${:.2f}".format)
                        st.dataframe(sell_df[["ticker", "current_price", "predicted_price", "expected_return", "recommendation"]], use_container_width=True)
if selected_tab == "Smart Dashboard":
    st.title("💰 Smart Financial Dashboard")
    st.subheader("Your Complete Financial Overview")

    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 📊 Financial Summary")
        with st.expander("View Complete Summary", expanded=True):
            total_expenses = sum(expense["amount"] for expense in st.session_state.financial_context.get("expenses", []))
            monthly_income = st.session_state.financial_context.get("budget", {}).get("income", 0)
            total_savings = monthly_income-total_expenses
            metric1, metric2, metric3 = st.columns(3)
            with metric1:
                st.metric("Total Savings", f"${total_savings:,.2f}")
            with metric2:
                st.metric("Monthly Income", f"${monthly_income:,.2f}")
            with metric3:
                st.metric("Monthly Expenses", f"${total_expenses:,.2f}")
            if monthly_income > 0:
                savings_rate = (total_savings / monthly_income) * 100
                st.progress(min(100, int(savings_rate)))
                st.caption(f"Savings Rate: {savings_rate:.1f}% of monthly income")
            if st.session_state.financial_context.get("goals"):
                st.markdown("### 🎯 Goals Progress")
                for goal in st.session_state.financial_context["goals"]:
                    target_date = datetime.strptime(goal["target_date"], "%Y-%m-%d")
                    days_remaining = (target_date - datetime.now()).days
                    progress = min(100, (total_savings / goal["target_amount"]) * 100) if goal["target_amount"] > 0 else 0
                    
                    st.write(f"**{goal['goal']}** - ${goal['target_amount']:,.2f}")
                    st.progress(progress/100)
                    st.caption(f"{days_remaining} days remaining - {progress:.1f}% complete")
    
    with col2:
        st.markdown("### 💡 Quick Insights")
        with st.expander("View Insights", expanded=True):
            if monthly_income > 0 and total_expenses > 0:
                expense_ratio = (total_expenses / monthly_income) * 100
                if expense_ratio > 80:
                    st.warning(f"⚠️ High expense ratio! You're spending {expense_ratio:.1f}% of your income")
                elif expense_ratio > 60:
                    st.info(f"ℹ️ Moderate expense ratio: {expense_ratio:.1f}% of income")
                else:
                    st.success(f"✅ Healthy expense ratio: {expense_ratio:.1f}% of income")
            
            if st.session_state.financial_context.get("portfolio"):
                portfolio_value = sum(stock["current_value"] for stock in st.session_state.financial_context["portfolio"])
                st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
    st.markdown("---")
    st.markdown("### 📈 Investment Report")
    
    if st.button("Generate Full Report"):
        with st.spinner("Generating comprehensive report..."):
            report_tab1, report_tab2, report_tab3 = st.tabs(["Portfolio Analysis", "Income vs Expenses", "Predictions"])
            
            with report_tab1:
                if st.session_state.financial_context.get("portfolio"):
                    st.markdown("#### Portfolio Performance")
                    portfolio = st.session_state.financial_context["portfolio"]
                    fig = go.Figure()
                    for stock in portfolio:
                        fig.add_trace(go.Bar(
                            x=[stock["ticker"]],
                            y=[stock["current_value"]],
                            name=stock["ticker"],
                            text=[f"${stock['current_value']:,.2f}<br>Return: {stock['profit_loss_percent']:.1f}%"],
                            textposition='auto'
                        ))
                    fig.update_layout(barmode='stack', title="Current Portfolio Value")
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("#### Individual Stock Performance")
                    for stock in portfolio:
                        col1, col2, col3 = st.columns([1,2,1])
                        with col1:
                            st.metric(stock["ticker"], f"${stock['current_price']:.2f}")
                        with col2:
                            st.progress(min(1.0, max(0.0, (stock["profit_loss_percent"] + 50)/100)))
                        with col3:
                            st.metric("Return", f"{stock['profit_loss_percent']:.1f}%", 
                                    f"${stock['profit_loss']:.2f}")
            
            with report_tab2:
                st.markdown("#### Income vs Expenses")
                if monthly_income > 0:
                    categories = {}
                    for expense in st.session_state.financial_context.get("expenses", []):
                        categories[expense["category"]] = categories.get(expense["category"], 0) + expense["amount"]
                    
                    if categories:
                        fig = go.Figure()
                        fig.add_trace(go.Pie(
                            labels=list(categories.keys()),
                            values=list(categories.values()),
                            hole=0.4,
                            textinfo='label+percent'
                        ))
                        fig.update_layout(title="Expense Breakdown")
                        st.plotly_chart(fig, use_container_width=True)
                        if "budgets" in st.session_state.financial_context.get("budget", {}):
                            budget_data = []
                            for category, budget in st.session_state.financial_context["budget"]["budgets"].items():
                                actual = categories.get(category, 0)
                                budget_data.append({
                                    "Category": category,
                                    "Budget": budget,
                                    "Actual": actual,
                                    "Difference": budget - actual
                                })
                            
                            budget_df = pd.DataFrame(budget_data)
                            st.dataframe(budget_df, use_container_width=True)
            
            with report_tab3:
                st.markdown("#### 1-Month Predictions")
                if "models" in st.session_state and "scalers" in st.session_state:
                    predictions = {}
                    for stock in st.session_state.financial_context.get("portfolio", []):
                        if stock["ticker"] in ["AAPL", "TSLA", "GOOGL", "NVDA"]:
                            pred = predict_next_month(
                                stock["ticker"], 
                                st.session_state.models, 
                                st.session_state.scalers
                            )
                            if pred is not None:
                                predictions[stock["ticker"]] = {
                                    "current": stock["current_price"],
                                    "predicted": pred[-1],
                                    "change": ((pred[-1] - stock["current_price"]) / stock["current_price"]) * 100
                                }
                    
                    if predictions:
                        pred_df = pd.DataFrame.from_dict(predictions, orient='index')
                        pred_df.columns = ["Current Price", "Predicted Price", "% Change"]
                        st.dataframe(pred_df.style.format({
                            "Current Price": "${:.2f}",
                            "Predicted Price": "${:.2f}",
                            "% Change": "{:.1f}%"
                        }), use_container_width=True)
                        fig = go.Figure()
                        for ticker, data in predictions.items():
                            fig.add_trace(go.Indicator(
                                mode="number+delta",
                                value=data["predicted"],
                                number={"prefix": "$"},
                                delta={
                                    "reference": data["current"],
                                    "relative": True,
                                    "valueformat": ".1%",
                                    "increasing": {"color": "green"},
                                    "decreasing": {"color": "red"}
                                },
                                title=ticker
                            ))
                        fig.update_layout(grid={"rows": 1, "columns": len(predictions)})
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Prediction models not loaded")
elif selected_tab == "Fraud Detection":
    st.title("Fraud Detection")

    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Add New Transaction")
        with st.form("transaction_form"):
            date = st.date_input("Transaction Date", datetime.now())
            time = st.time_input("Transaction Time", datetime.now().time())
            amount = st.number_input("Amount", min_value=0.01, step=0.01)
            merchant = st.text_input("Merchant Name")
            category = st.selectbox(
                "Category", 
                ["Groceries", "Dining", "Shopping", "Entertainment", "Travel", "Services", "Other"]
            )
            location = st.text_input("Location (City, State)")
            location_usual = st.checkbox("Is this your usual location?", value=True)
            category_usual = st.checkbox("Is this a usual merchant category for you?", value=True)
            time_usual = st.checkbox("Is this a usual transaction time for you?", value=True)
            
            submitted = st.form_submit_button("Add Transaction")
            
            if submitted:
                transaction = {
                    "date": date.strftime("%Y-%m-%d"),
                    "time": time.strftime("%H:%M:%S"),
                    "amount": amount,
                    "merchant": merchant,
                    "category": category,
                    "location": location,
                    "location_usual": location_usual,
                    "category_usual": category_usual,
                    "time_usual": time_usual,
                    "rapid_succession": False  
                }
                
                
                recent_transactions = [
                    t for t in st.session_state.transactions
                    if datetime.strptime(t['date'], "%Y-%m-%d").date() >= (date - timedelta(days=1))
                ]
                if len(recent_transactions) >= 3:
                    transaction["rapid_succession"] = True
                
                st.session_state.transactions.append(transaction)
                st.success("Transaction added successfully!")
    
    with col2:
        st.subheader("Fraud Detection Settings")
        
        # Fraud detection settings
        detection_threshold = st.slider(
            "Detection Sensitivity", 
            min_value=30, 
            max_value=70, 
            value=40,
            help="Lower values will flag more transactions as potentially fraudulent"
        )
        if st.button("Run Fraud Detection"):
            with st.spinner("Analyzing transactions for potential fraud..."):
                fraud_indicators = detect_fraud_in_transactions(st.session_state.transactions)
                
                if fraud_indicators:
                    st.warning(f"Detected {len(fraud_indicators)} potentially fraudulent transactions")
                else:
                    st.success("No potentially fraudulent transactions detected")
    st.subheader("Transaction History")
    
    if not st.session_state.transactions:
        st.info("No transactions recorded yet. Add transactions using the form above.")
    else:
        df_transactions = pd.DataFrame(st.session_state.transactions)
        if 'fraud_score' not in df_transactions.columns:
            df_transactions['fraud_score'] = 0
        def highlight_fraud(row):
            if row['fraud_score'] >= 40:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        st.dataframe(
            df_transactions.style.apply(highlight_fraud, axis=1),
            use_container_width=True
        )
    if len(st.session_state.transactions) >= 5:
        st.subheader("Fraud Pattern Visualization")
        viz_tab1, viz_tab2 = st.tabs(["Transaction Amounts", "Categories"])
        
        with viz_tab1:
            df = pd.DataFrame(st.session_state.transactions)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['amount'],
                    mode='markers',
                    name='Transaction Amount',
                    marker=dict(
                        size=10,
                        color=df['fraud_score'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Fraud Score")
                    )
                ),
                secondary_y=False,
            )
            if len(df) > 3:
                df = df.sort_values('date')
                df['rolling_avg'] = df['amount'].rolling(window=3).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['rolling_avg'],
                        mode='lines',
                        name='3-Day Rolling Average',
                        line=dict(color='red', width=2)
                    ),
                    secondary_y=False,
                )

            fig.update_layout(
                title_text="Transaction Amounts Over Time",
                xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Amount", secondary_y=False)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            category_counts = df['category'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            category_fraud = df.groupby('category')['fraud_score'].mean().reset_index()
            category_fraud.columns = ['category', 'avg_fraud_score']
            category_data = pd.merge(category_counts, category_fraud, on='category')
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=category_data['category'],
                y=category_data['count'],
                marker=dict(
                    color=category_data['avg_fraud_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Fraud Score")
                ),
                text=category_data['avg_fraud_score'].round(1),
                textposition='auto',
            ))
            
            fig.update_layout(
                title_text="Transaction Categories with Fraud Risk",
                xaxis_title="Category",
                yaxis_title="Number of Transactions"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Fraud Insights")
    
    if len(st.session_state.transactions) > 0:
        fraud_indicators = detect_fraud_in_transactions(st.session_state.transactions)
        
        if fraud_indicators:
            for i, transaction in enumerate(fraud_indicators):
                with st.expander(f"Suspicious Transaction: {transaction['merchant']} - ${transaction['amount']:.2f} on {transaction['date']}"):
                    st.write(f"**Fraud Score:** {transaction['fraud_score']}/100")
                    st.write("**Reasons flagged:**")
                    for reason in transaction['fraud_reasons']:
                        st.write(f"- {reason}")
                    st.write("**Transaction Details:**")
                    for key, value in transaction.items():
                        if key not in ['fraud_score', 'fraud_reasons']:
                            st.write(f"- {key.replace('_', ' ').title()}: {value}")
                    st.write("**Recommendation:**")
                    if transaction['fraud_score'] > 60:
                        st.error("High risk transaction. Consider contacting your bank immediately.")
                    elif transaction['fraud_score'] > 40:
                        st.warning("Moderate risk. Verify this transaction with your records.")
                    else:
                        st.info("Low risk. Normal transaction pattern.")
        else:
            st.info("No suspicious transactions detected in your transaction history.")
    
