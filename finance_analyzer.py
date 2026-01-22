import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class FinanceAnalyzer:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
    
    def generate_budget_summary(self, expenses_data, model_choice="gemini"):
        """Generate AI-powered budget summary"""
        
        # Process the data
        if isinstance(expenses_data, str):
            # If it's text input
            data_text = expenses_data
        elif isinstance(expenses_data, pd.DataFrame):
            # If it's a DataFrame
            data_text = expenses_data.to_string()
        else:
            data_text = str(expenses_data)
        
        prompt = f"""
        As a personal finance advisor, analyze this spending data and create a comprehensive budget summary:

        SPENDING DATA:
        {data_text}

        Please provide:
        1. 📊 TOTAL SPENDING ANALYSIS
        2. 📈 CATEGORY BREAKDOWN (percentages)
        3. 💡 SAVINGS OPPORTUNITIES
        4. ⚠️  AREAS OF CONCERN
        5. 🎯 ACTIONABLE RECOMMENDATIONS

        Keep it concise but actionable. Use emojis and bullet points for better readability.
        """
        
        return self.llm_manager.get_response(model_choice, prompt)
    
    def get_spending_insights(self, transactions, model_choice="granite_instruct"):
        """Get spending pattern insights"""
        
        prompt = f"""
        Analyze these financial transactions and provide intelligent insights:

        TRANSACTIONS:
        {transactions}

        Focus on:
        1. 🔍 SPENDING PATTERNS (daily, weekly, monthly trends)
        2. 🏷️  TOP SPENDING CATEGORIES
        3. 📅 TIMING PATTERNS (when do they spend most?)
        4. 💳 UNUSUAL OR LARGE TRANSACTIONS
        5. 🎯 PERSONALIZED SUGGESTIONS FOR IMPROVEMENT

        Be specific and actionable in your recommendations.
        """
        
        return self.llm_manager.get_response(model_choice, prompt)
    
    def create_spending_visualization(self, data):
        """Create interactive spending charts"""
        if isinstance(data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame(list(data.items()), columns=['Category', 'Amount'])
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return None
        
        # Create pie chart
        fig_pie = px.pie(
            df, 
            values='Amount', 
            names='Category',
            title='💰 Spending by Category',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Create bar chart
        fig_bar = px.bar(
            df,
            x='Category',
            y='Amount',
            title='📊 Category-wise Spending',
            color='Amount',
            color_continuous_scale='viridis'
        )
        
        return fig_pie, fig_bar
    
    def generate_savings_plan(self, income, expenses, goals, model_choice="gemini"):
        """Generate personalized savings plan"""
        
        prompt = f"""
        Create a personalized savings plan based on this financial information:

        💰 MONTHLY INCOME: ${income}
        💸 MONTHLY EXPENSES: ${expenses}
        🎯 FINANCIAL GOALS: {goals}

        Please create:
        1. 📋 SAVINGS CAPACITY ANALYSIS
        2. 🎯 GOAL-BASED ALLOCATION STRATEGY
        3. 📅 TIMELINE FOR EACH GOAL
        4. 💡 PRACTICAL TIPS TO INCREASE SAVINGS
        5. 🚨 EMERGENCY FUND RECOMMENDATIONS

        Make it actionable with specific dollar amounts and timelines.
        """
        
        return self.llm_manager.get_response(model_choice, prompt)