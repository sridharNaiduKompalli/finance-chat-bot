# utils/visualizations.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st

class FinancialVisualizer:
    """
    Comprehensive visualization utilities for financial data analysis
    """
    
    def __init__(self):
        self.color_schemes = {
            'default': px.colors.qualitative.Set3,
            'expenses': px.colors.sequential.Reds,
            'income': px.colors.sequential.Greens,
            'budget': px.colors.qualitative.Pastel,
            'trend': px.colors.sequential.Viridis
        }
        
        self.chart_templates = {
            'clean': 'plotly_white',
            'dark': 'plotly_dark',
            'minimal': 'simple_white'
        }
    
    def create_spending_pie_chart(self, df: pd.DataFrame, title: str = "💰 Spending by Category") -> go.Figure:
        """Create an interactive pie chart for spending categories"""
        if 'category' not in df.columns or 'amount' not in df.columns:
            return self._create_error_chart("Missing required columns: category, amount")
        
        category_totals = df.groupby('category')['amount'].sum().reset_index()
        category_totals = category_totals.sort_values('amount', ascending=False)
        
        fig = px.pie(
            category_totals,
            values='amount',
            names='category',
            title=title,
            color_discrete_sequence=self.color_schemes['default']
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            font_size=12,
            title_font_size=16,
            showlegend=True,
            template=self.chart_templates['clean']
        )
        
        return fig
    
    def create_spending_bar_chart(self, df: pd.DataFrame, title: str = "📊 Category-wise Spending") -> go.Figure:
        """Create a horizontal bar chart for spending categories"""
        if 'category' not in df.columns or 'amount' not in df.columns:
            return self._create_error_chart("Missing required columns: category, amount")
        
        category_totals = df.groupby('category')['amount'].sum().reset_index()
        category_totals = category_totals.sort_values('amount', ascending=True)  # Ascending for horizontal bars
        
        fig = px.bar(
            category_totals,
            x='amount',
            y='category',
            orientation='h',
            title=title,
            color='amount',
            color_continuous_scale='viridis'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Amount: $%{x:,.2f}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title="Amount ($)",
            yaxis_title="Category",
            template=self.chart_templates['clean'],
            coloraxis_colorbar_title="Amount"
        )
        
        return fig
    
    def create_time_series_chart(self, df: pd.DataFrame, title: str = "📈 Spending Over Time") -> go.Figure:
        """Create a time series chart of spending patterns"""
        if 'date' not in df.columns or 'amount' not in df.columns:
            return self._create_error_chart("Missing required columns: date, amount")
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        df_sorted = df.sort_values('date')
        
        # Group by date for daily spending
        daily_spending = df_sorted.groupby('date')['amount'].sum().reset_index()
        
        fig = px.line(
            daily_spending,
            x='date',
            y='amount',
            title=title,
            line_shape='spline'
        )
        
        fig.update_traces(
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>%{x}</b><br>Total Spending: $%{y:,.2f}<extra></extra>'
        )
        
        # Add trend line
        if len(daily_spending) > 5:
            z = np.polyfit(range(len(daily_spending)), daily_spending['amount'], 1)
            p = np.poly1d(z)
            trend_line = p(range(len(daily_spending)))
            
            fig.add_scatter(
                x=daily_spending['date'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>Trend</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
            )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            template=self.chart_templates['clean'],
            hovermode='x unified'
        )
        
        return fig
    
    def create_monthly_spending_chart(self, df: pd.DataFrame, title: str = "📅 Monthly Spending Breakdown") -> go.Figure:
        """Create a monthly spending breakdown chart"""
        if 'date' not in df.columns or 'amount' not in df.columns:
            return self._create_error_chart("Missing required columns: date, amount")
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').astype(str)
        
        if 'category' in df.columns:
            # Stacked bar chart by category
            monthly_category = df.groupby(['month', 'category'])['amount'].sum().reset_index()
            
            fig = px.bar(
                monthly_category,
                x='month',
                y='amount',
                color='category',
                title=title,
                color_discrete_sequence=self.color_schemes['default']
            )
        else:
            # Simple monthly totals
            monthly_totals = df.groupby('month')['amount'].sum().reset_index()
            
            fig = px.bar(
                monthly_totals,
                x='month',
                y='amount',
                title=title,
                color='amount',
                color_continuous_scale='blues'
            )
        
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Amount: $%{y:,.2f}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            template=self.chart_templates['clean']
        )
        
        return fig
    
    def create_budget_comparison_chart(self, actual_df: pd.DataFrame, budget_dict: Dict[str, float], 
                                     title: str = "💼 Budget vs Actual Spending") -> go.Figure:
        """Create a budget comparison chart"""
        if 'category' not in actual_df.columns or 'amount' not in actual_df.columns:
            return self._create_error_chart("Missing required columns: category, amount")
        
        actual_spending = actual_df.groupby('category')['amount'].sum().to_dict()
        
        categories = list(set(list(actual_spending.keys()) + list(budget_dict.keys())))
        budget_amounts = [budget_dict.get(cat, 0) for cat in categories]
        actual_amounts = [actual_spending.get(cat, 0) for cat in categories]
        
        fig = go.Figure()
        
        # Budget bars
        fig.add_trace(go.Bar(
            name='Budget',
            x=categories,
            y=budget_amounts,
            marker_color='lightblue',
            hovertemplate='<b>%{x}</b><br>Budget: $%{y:,.2f}<extra></extra>'
        ))
        
        # Actual spending bars
        fig.add_trace(go.Bar(
            name='Actual',
            x=categories,
            y=actual_amounts,
            marker_color='darkblue',
            hovertemplate='<b>%{x}</b><br>Actual: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Category",
            yaxis_title="Amount ($)",
            barmode='group',
            template=self.chart_templates['clean']
        )
        
        return fig
    
    def create_spending_heatmap(self, df: pd.DataFrame, title: str = "🔥 Spending Heatmap") -> go.Figure:
        """Create a heatmap showing spending patterns by day of week and hour"""
        if 'date' not in df.columns or 'amount' not in df.columns:
            return self._create_error_chart("Missing required columns: date, amount")
        
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        
        # Create pivot table for heatmap
        heatmap_data = df.groupby(['day_of_week', 'hour'])['amount'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='amount').fillna(0)
        
        # Reorder days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(day_order)
        
        fig = px.imshow(
            heatmap_pivot,
            title=title,
            color_continuous_scale='Reds',
            aspect='auto'
        )
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            template=self.chart_templates['clean']
        )
        
        return fig
    
    def create_top_transactions_chart(self, df: pd.DataFrame, n: int = 10, 
                                    title: str = "💸 Top Transactions") -> go.Figure:
        """Create a chart showing the top N transactions"""
        if 'amount' not in df.columns:
            return self._create_error_chart("Missing required column: amount")
        
        top_transactions = df.nlargest(n, 'amount').copy()
        
        # Create labels for transactions
        if 'description' in top_transactions.columns:
            top_transactions['label'] = top_transactions['description'].str[:30] + '...'
        else:
            top_transactions['label'] = [f'Transaction {i+1}' for i in range(len(top_transactions))]
        
        fig = px.bar(
            top_transactions,
            x='label',
            y='amount',
            title=title,
            color='amount',
            color_continuous_scale='Reds'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Amount: $%{y:,.2f}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title="Transaction",
            yaxis_title="Amount ($)",
            template=self.chart_templates['clean'],
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_savings_goal_chart(self, current_savings: float, goal_amount: float, 
                                monthly_contribution: float, title: str = "🎯 Savings Goal Progress") -> go.Figure:
        """Create a savings goal progress chart"""
        progress_percentage = (current_savings / goal_amount) * 100 if goal_amount > 0 else 0
        remaining = max(0, goal_amount - current_savings)
        months_to_goal = remaining / monthly_contribution if monthly_contribution > 0 else float('inf')
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=progress_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': 100, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        # Add annotations
        fig.add_annotation(
            text=f"Current: ${current_savings:,.2f}<br>Goal: ${goal_amount:,.2f}<br>Remaining: ${remaining:,.2f}",
            x=0.5, y=0.2,
            showarrow=False,
            font=dict(size=12)
        )
        
        if months_to_goal != float('inf'):
            fig.add_annotation(
                text=f"Months to goal: {months_to_goal:.1f}",
                x=0.5, y=0.1,
                showarrow=False,
                font=dict(size=10, color="gray")
            )
        
        fig.update_layout(
            template=self.chart_templates['clean'],
            height=400
        )
        
        return fig
    
    def create_dashboard_summary(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create a comprehensive dashboard with multiple charts"""
        charts = {}
        
        if df.empty:
            error_fig = self._create_error_chart("No data available")
            return {'error': error_fig}
        
        # Main spending charts
        charts['pie_chart'] = self.create_spending_pie_chart(df)
        charts['bar_chart'] = self.create_spending_bar_chart(df)
        
        # Time-based charts
        if 'date' in df.columns:
            charts['time_series'] = self.create_time_series_chart(df)
            charts['monthly_chart'] = self.create_monthly_spending_chart(df)
        
        # Top transactions
        charts['top_transactions'] = self.create_top_transactions_chart(df)
        
        return charts
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create a chart displaying an error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ {error_message}",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template=self.chart_templates['clean'],
            height=400
        )
        return fig
    
    def export_chart(self, fig: go.Figure, filename: str, format: str = 'html') -> str:
        """Export chart to various formats"""
        try:
            if format.lower() == 'html':
                fig.write_html(filename)
            elif format.lower() == 'png':
                fig.write_image(filename)
            elif format.lower() == 'pdf':
                fig.write_image(filename)
            else:
                return f"Unsupported format: {format}"
            
            return f"Chart exported successfully to {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"
    
    def apply_custom_theme(self, fig: go.Figure, theme: str = 'clean') -> go.Figure:
        """Apply custom themes to charts"""
        if theme in self.chart_templates:
            fig.update_layout(template=self.chart_templates[theme])
        
        # Add custom styling based on theme
        if theme == 'dark':
            fig.update_layout(
                plot_bgcolor='rgb(17, 17, 17)',
                paper_bgcolor='rgb(17, 17, 17)',
                font_color='white'
            )
        elif theme == 'minimal':
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12, color="black")
            )
        
        return fig
    
    def create_spending_treemap(self, df: pd.DataFrame, title: str = "🌳 Spending Treemap") -> go.Figure:
        """Create a treemap visualization of spending categories"""
        if 'category' not in df.columns or 'amount' not in df.columns:
            return self._create_error_chart("Missing required columns: category, amount")
        
        category_totals = df.groupby('category')['amount'].sum().reset_index()
        
        fig = px.treemap(
            category_totals,
            path=['category'],
            values='amount',
            title=title,
            color='amount',
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percentParent}<extra></extra>'
        )
        
        fig.update_layout(
            template=self.chart_templates['clean']
        )
        
        return fig
    
    def create_waterfall_chart(self, income: float, expenses_dict: Dict[str, float], 
                              title: str = "💧 Cash Flow Waterfall") -> go.Figure:
        """Create a waterfall chart showing cash flow"""
        categories = ['Income'] + list(expenses_dict.keys()) + ['Net']
        values = [income] + [-abs(v) for v in expenses_dict.values()] + [income - sum(expenses_dict.values())]
        
        # Create cumulative values for waterfall effect
        cumulative = [income]
        for expense in expenses_dict.values():
            cumulative.append(cumulative[-1] - expense)
        
        fig = go.Figure()
        
        # Add income bar
        fig.add_trace(go.Bar(
            name='Income',
            x=[categories[0]],
            y=[values[0]],
            marker_color='green',
            hovertemplate='<b>%{x}</b><br>Amount: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add expense bars
        for i, (category, value) in enumerate(expenses_dict.items(), 1):
            fig.add_trace(go.Bar(
                name=category,
                x=[category],
                y=[abs(value)],
                base=[cumulative[i]],
                marker_color='red',
                hovertemplate=f'<b>{category}</b><br>Expense: $%{{y:,.2f}}<extra></extra>'
            ))
        
        # Add net bar
        net_value = values[-1]
        fig.add_trace(go.Bar(
            name='Net',
            x=[categories[-1]],
            y=[abs(net_value)],
            marker_color='green' if net_value >= 0 else 'red',
            hovertemplate='<b>Net</b><br>Amount: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Category",
            yaxis_title="Amount ($)",
            showlegend=False,
            template=self.chart_templates['clean']
        )
        
        return fig
    
    def create_category_trend_chart(self, df: pd.DataFrame, title: str = "📊 Category Trends Over Time") -> go.Figure:
        """Create a multi-line chart showing category spending trends"""
        if 'date' not in df.columns or 'category' not in df.columns or 'amount' not in df.columns:
            return self._create_error_chart("Missing required columns: date, category, amount")
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').astype(str)
        
        monthly_category = df.groupby(['month', 'category'])['amount'].sum().reset_index()
        
        fig = px.line(
            monthly_category,
            x='month',
            y='amount',
            color='category',
            title=title,
            markers=True
        )
        
        fig.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>Amount: $%{y:,.2f}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Amount ($)", 
            template=self.chart_templates['clean'],
            hovermode='x unified'
        )
        
        return fig
    
    def create_spending_distribution_chart(self, df: pd.DataFrame, title: str = "📈 Spending Distribution") -> go.Figure:
        """Create a histogram showing spending amount distribution"""
        if 'amount' not in df.columns:
            return self._create_error_chart("Missing required column: amount")
        
        fig = px.histogram(
            df,
            x='amount',
            nbins=30,
            title=title,
            color_discrete_sequence=['skyblue']
        )
        
        fig.update_traces(
            hovertemplate='<b>Amount Range</b><br>$%{x}<br>Count: %{y}<extra></extra>'
        )
        
        # Add statistical lines
        mean_amount = df['amount'].mean()
        median_amount = df['amount'].median()
        
        fig.add_vline(
            x=mean_amount,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mean_amount:.2f}"
        )
        
        fig.add_vline(
            x=median_amount,
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Median: ${median_amount:.2f}"
        )
        
        fig.update_layout(
            xaxis_title="Transaction Amount ($)",
            yaxis_title="Frequency",
            template=self.chart_templates['clean']
        )
        
        return fig
    
    def create_budget_gauge_dashboard(self, budget_data: Dict[str, Dict[str, float]], 
                                    title: str = "🎯 Budget Performance Dashboard") -> go.Figure:
        """Create a dashboard with multiple gauges for different budget categories"""
        categories = list(budget_data.keys())
        n_categories = len(categories)
        
        # Calculate grid dimensions
        cols = min(3, n_categories)
        rows = (n_categories + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
            subplot_titles=categories,
            vertical_spacing=0.1
        )
        
        for i, (category, data) in enumerate(budget_data.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            actual = data.get('actual', 0)
            budget = data.get('budget', 1)
            percentage = (actual / budget) * 100 if budget > 0 else 0
            
            # Determine color based on performance
            if percentage <= 80:
                color = "green"
            elif percentage <= 100:
                color = "yellow"
            else:
                color = "red"
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    title={'text': f"{category}<br>${actual:.0f} / ${budget:.0f}"},
                    gauge={
                        'axis': {'range': [None, 150]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 80], 'color': "lightgreen"},
                            {'range': [80, 100], 'color': "yellow"},
                            {'range': [100, 150], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 100
                        }
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            template=self.chart_templates['clean'],
            height=300 * rows
        )
        
        return fig