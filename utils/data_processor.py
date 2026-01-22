# utils/data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional, Union
import json

class FinancialDataProcessor:
    """
    Comprehensive data processing utilities for financial data
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'xlsx', 'txt']
        self.common_categories = {
            'food': ['grocery', 'restaurant', 'food', 'dining', 'starbucks', 'mcdonalds'],
            'transportation': ['gas', 'uber', 'taxi', 'bus', 'train', 'parking', 'car'],
            'entertainment': ['movie', 'netflix', 'spotify', 'game', 'concert', 'theater'],
            'shopping': ['amazon', 'target', 'walmart', 'mall', 'store', 'shopping'],
            'utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'utility'],
            'healthcare': ['doctor', 'pharmacy', 'hospital', 'medical', 'dentist'],
            'education': ['school', 'tuition', 'books', 'course', 'training'],
            'housing': ['rent', 'mortgage', 'repair', 'maintenance', 'home'],
            'misc': ['atm', 'fee', 'charge', 'other', 'miscellaneous']
        }
    
    def clean_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize financial data
        """
        df_clean = df.copy()
        
        # Standard column name mapping
        column_mapping = {
            'amt': 'amount', 'price': 'amount', 'cost': 'amount', 'value': 'amount',
            'desc': 'description', 'details': 'description', 'memo': 'description',
            'cat': 'category', 'type': 'category', 'group': 'category',
            'dt': 'date', 'transaction_date': 'date', 'timestamp': 'date'
        }
        
        # Rename columns to standard format
        df_clean.columns = df_clean.columns.str.lower().str.strip()
        df_clean.rename(columns=column_mapping, inplace=True)
        
        # Clean amount column
        if 'amount' in df_clean.columns:
            df_clean['amount'] = self._clean_amount_column(df_clean['amount'])
        
        # Clean date column
        if 'date' in df_clean.columns:
            df_clean['date'] = self._clean_date_column(df_clean['date'])
        
        # Clean description column
        if 'description' in df_clean.columns:
            df_clean['description'] = self._clean_text_column(df_clean['description'])
        
        # Auto-categorize transactions if category is missing
        if 'description' in df_clean.columns and 'category' not in df_clean.columns:
            df_clean['category'] = df_clean['description'].apply(self._auto_categorize)
        
        # Remove duplicates and null amounts
        df_clean = df_clean.drop_duplicates()
        if 'amount' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['amount'])
        
        return df_clean
    
    def _clean_amount_column(self, amount_series: pd.Series) -> pd.Series:
        """Clean and convert amount column to numeric"""
        # Convert to string first
        amount_clean = amount_series.astype(str)
        # Remove currency symbols and commas
        amount_clean = amount_clean.str.replace(r'[$,€£¥₹]', '', regex=True)
        # Remove parentheses (often used for negative amounts)
        amount_clean = amount_clean.str.replace(r'[()]', '', regex=True)
        # Convert to numeric, coerce errors to NaN
        amount_clean = pd.to_numeric(amount_clean, errors='coerce')
        return amount_clean.abs()  # Use absolute values for consistency
    
    def _clean_date_column(self, date_series: pd.Series) -> pd.Series:
        """Clean and standardize date column"""
        return pd.to_datetime(date_series, errors='coerce', infer_datetime_format=True)
    
    def _clean_text_column(self, text_series: pd.Series) -> pd.Series:
        """Clean text columns"""
        text_clean = text_series.astype(str)
        text_clean = text_clean.str.strip().str.lower()
        # Remove extra whitespace
        text_clean = text_clean.str.replace(r'\s+', ' ', regex=True)
        return text_clean
    
    def _auto_categorize(self, description: str) -> str:
        """Automatically categorize transactions based on description"""
        if pd.isna(description) or description == '':
            return 'misc'
        
        description_lower = str(description).lower()
        
        for category, keywords in self.common_categories.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return 'misc'
    
    def calculate_spending_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive spending metrics"""
        metrics = {}
        
        if 'amount' not in df.columns:
            return {'error': 'Amount column not found'}
        
        # Basic metrics
        metrics['total_spending'] = df['amount'].sum()
        metrics['average_transaction'] = df['amount'].mean()
        metrics['median_transaction'] = df['amount'].median()
        metrics['transaction_count'] = len(df)
        
        # Category analysis
        if 'category' in df.columns:
            category_summary = df.groupby('category')['amount'].agg(['sum', 'count', 'mean']).round(2)
            metrics['category_breakdown'] = category_summary.to_dict('index')
            metrics['top_spending_category'] = df.groupby('category')['amount'].sum().idxmax()
        
        # Date-based analysis
        if 'date' in df.columns:
            df_dated = df.dropna(subset=['date'])
            if not df_dated.empty:
                metrics['date_range'] = {
                    'start': df_dated['date'].min().strftime('%Y-%m-%d'),
                    'end': df_dated['date'].max().strftime('%Y-%m-%d')
                }
                
                # Monthly spending
                df_dated['month'] = df_dated['date'].dt.to_period('M')
                monthly_spending = df_dated.groupby('month')['amount'].sum()
                metrics['monthly_average'] = monthly_spending.mean()
                metrics['monthly_spending'] = monthly_spending.to_dict()
        
        # Spending patterns
        metrics['large_transactions'] = len(df[df['amount'] > df['amount'].quantile(0.9)])
        metrics['small_transactions'] = len(df[df['amount'] < df['amount'].quantile(0.1)])
        
        return metrics
    
    def detect_spending_anomalies(self, df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """Detect unusual spending patterns using statistical methods"""
        if 'amount' not in df.columns:
            return pd.DataFrame()
        
        # Calculate Z-scores
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        df['z_score'] = (df['amount'] - mean_amount) / std_amount
        
        # Identify anomalies
        anomalies = df[abs(df['z_score']) > threshold].copy()
        anomalies['anomaly_type'] = anomalies['z_score'].apply(
            lambda x: 'unusually_high' if x > threshold else 'unusually_low'
        )
        
        return anomalies[['date', 'description', 'amount', 'category', 'anomaly_type']] if not anomalies.empty else pd.DataFrame()
    
    def create_budget_comparison(self, actual_df: pd.DataFrame, budget_dict: Dict[str, float]) -> pd.DataFrame:
        """Compare actual spending with budget"""
        if 'category' not in actual_df.columns or 'amount' not in actual_df.columns:
            return pd.DataFrame()
        
        actual_spending = actual_df.groupby('category')['amount'].sum().to_dict()
        
        comparison_data = []
        for category, budgeted in budget_dict.items():
            actual = actual_spending.get(category, 0)
            difference = actual - budgeted
            percentage = (actual / budgeted * 100) if budgeted > 0 else 0
            
            comparison_data.append({
                'category': category,
                'budgeted': budgeted,
                'actual': actual,
                'difference': difference,
                'percentage_of_budget': percentage,
                'status': 'over' if difference > 0 else 'under'
            })
        
        return pd.DataFrame(comparison_data)
    
    def generate_spending_insights(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate natural language insights about spending patterns"""
        insights = {}
        
        if df.empty or 'amount' not in df.columns:
            return {'error': 'Insufficient data for insights'}
        
        # Total spending insight
        total = df['amount'].sum()
        insights['total_spending'] = f"💰 Total spending: ${total:,.2f} across {len(df)} transactions"
        
        # Category insights
        if 'category' in df.columns:
            top_category = df.groupby('category')['amount'].sum().idxmax()
            top_amount = df.groupby('category')['amount'].sum().max()
            insights['top_category'] = f"🏆 Highest spending category: {top_category.title()} (${top_amount:,.2f})"
        
        # Transaction frequency
        avg_transaction = df['amount'].mean()
        insights['avg_transaction'] = f"📊 Average transaction: ${avg_transaction:.2f}"
        
        # Large transactions
        large_threshold = df['amount'].quantile(0.9)
        large_transactions = len(df[df['amount'] > large_threshold])
        insights['large_transactions'] = f"⚠️  Large transactions (top 10%): {large_transactions} transactions above ${large_threshold:.2f}"
        
        # Date-based insights
        if 'date' in df.columns:
            df_dated = df.dropna(subset=['date'])
            if not df_dated.empty:
                date_range = (df_dated['date'].max() - df_dated['date'].min()).days
                insights['time_period'] = f"📅 Data covers {date_range} days of spending"
        
        return insights
    
    def export_processed_data(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> str:
        """Export processed data to various formats"""
        try:
            if format.lower() == 'csv':
                df.to_csv(filename, index=False)
            elif format.lower() == 'json':
                df.to_json(filename, orient='records', indent=2)
            elif format.lower() == 'xlsx':
                df.to_excel(filename, index=False)
            else:
                return f"Unsupported format: {format}"
            
            return f"Data exported successfully to {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"
    
    def create_sample_data(self, num_transactions: int = 50) -> pd.DataFrame:
        """Generate sample financial data for testing"""
        categories = list(self.common_categories.keys())
        
        # Generate random dates over the last 3 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        sample_data = []
        for i in range(num_transactions):
            # Random amount with realistic distribution
            if np.random.random() < 0.7:  # 70% small transactions
                amount = np.random.uniform(5, 100)
            elif np.random.random() < 0.9:  # 20% medium transactions
                amount = np.random.uniform(100, 500)
            else:  # 10% large transactions
                amount = np.random.uniform(500, 2000)
            
            category = np.random.choice(categories)
            date = np.random.choice(date_range)
            
            # Generate description based on category
            descriptions = {
                'food': ['Grocery Store', 'Restaurant XYZ', 'Coffee Shop', 'Fast Food'],
                'transportation': ['Gas Station', 'Uber Ride', 'Parking Fee', 'Bus Pass'],
                'entertainment': ['Movie Theater', 'Streaming Service', 'Concert Ticket', 'Gaming'],
                'shopping': ['Amazon Purchase', 'Department Store', 'Online Shopping', 'Mall'],
                'utilities': ['Electric Bill', 'Internet Bill', 'Phone Bill', 'Water Bill'],
                'healthcare': ['Pharmacy', 'Doctor Visit', 'Dental Checkup', 'Medical'],
                'education': ['Course Fee', 'Book Store', 'Training Program', 'School'],
                'housing': ['Rent Payment', 'Home Repair', 'Maintenance', 'Property'],
                'misc': ['ATM Fee', 'Bank Charge', 'Other Expense', 'Miscellaneous']
            }
            
            description = np.random.choice(descriptions.get(category, ['General Expense']))
            
            sample_data.append({
                'date': date,
                'description': description,
                'amount': round(amount, 2),
                'category': category
            })
        
        return pd.DataFrame(sample_data).sort_values('date').reset_index(drop=True)