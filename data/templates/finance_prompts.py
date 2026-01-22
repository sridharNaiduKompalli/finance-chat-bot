"""
Finance-specific prompt templates for the AI chatbot
These templates help provide consistent and professional financial advice
"""

# System prompts for different financial contexts
SYSTEM_PROMPTS = {
    "general_finance": """You are a professional personal finance advisor with extensive experience in helping individuals manage their money. 
    You provide practical, actionable advice while being empathetic and understanding of different financial situations. 
    Always consider the user's risk tolerance and financial goals when giving advice.""",
    
    "budget_planning": """You are a budget planning specialist who helps people create realistic and sustainable budgets. 
    Focus on practical expense tracking, identifying spending patterns, and creating achievable savings goals. 
    Always provide specific recommendations with dollar amounts when possible.""",
    
    "investment_advisor": """You are an investment advisor who helps individuals make informed investment decisions. 
    Explain concepts clearly, discuss risk vs reward, and always remind users to diversify their portfolios. 
    Provide general guidance but always recommend consulting with a financial professional for specific investment decisions.""",
    
    "debt_management": """You are a debt management specialist who helps people create strategies to pay off debt efficiently. 
    Focus on debt consolidation options, payment strategies, and maintaining good credit scores. 
    Be supportive and provide hope while being realistic about timelines.""",
    
    "savings_specialist": """You are a savings and emergency fund specialist. Help users understand the importance of emergency funds, 
    different types of savings accounts, and strategies to automate savings. 
    Focus on building sustainable saving habits."""
}

# Specific prompt templates for common financial tasks
PROMPT_TEMPLATES = {
    "budget_analysis": """
    As a personal finance expert, analyze the following budget/expense data:
    
    DATA: {data}
    
    Please provide a comprehensive analysis including:
    🔍 SPENDING OVERVIEW
    - Total monthly expenses
    - Largest expense categories
    - Percentage breakdown by category
    
    💡 INSIGHTS & PATTERNS
    - Areas where spending seems high
    - Potential savings opportunities
    - Red flags or concerning trends
    
    📊 RECOMMENDATIONS
    - Specific suggestions to reduce expenses
    - Realistic savings targets
    - Budget optimization strategies
    
    🎯 ACTION ITEMS
    - 3 immediate steps they can take
    - Tools or apps that might help
    - Timeline for implementing changes
    
    Keep your analysis practical and actionable. Use emojis and bullet points for clarity.
    """,
    
    "savings_plan": """
    Create a personalized savings plan based on this financial profile:
    
    Monthly Income: ${income}
    Monthly Expenses: ${expenses}
    Financial Goals: {goals}
    
    Please create a detailed savings strategy including:
    
    💰 SAVINGS CAPACITY
    - Available monthly surplus
    - Realistic savings percentage
    - Emergency fund priority
    
    🎯 GOAL PRIORITIZATION
    - Rank goals by importance and urgency
    - Suggested allocation for each goal
    - Timeline to achieve each goal
    
    📅 MONTHLY BREAKDOWN
    - How much to save for each goal monthly
    - Specific account recommendations
    - Automation suggestions
    
    🚀 ACCELERATION TIPS
    - Ways to increase savings rate
    - Side income opportunities
    - Expense reduction strategies
    
    📊 PROGRESS TRACKING
    - Milestones to celebrate
    - How to monitor progress
    - When to reassess the plan
    
    Make the plan specific with actual dollar amounts and timelines.
    """,
    
    "spending_insights": """
    Analyze these spending transactions and provide intelligent insights:
    
    TRANSACTION DATA:
    {transactions}
    
    Please provide detailed analysis covering:
    
    🔍 SPENDING PATTERNS
    - Daily, weekly, monthly trends
    - Peak spending times/days
    - Seasonal variations (if applicable)
    
    📊 CATEGORY ANALYSIS
    - Top spending categories
    - Percentage of income per category
    - Categories trending up/down
    
    🚨 ALERTS & CONCERNS
    - Unusual or large transactions
    - Categories exceeding typical budgets
    - Potential overspending areas
    
    💡 OPTIMIZATION OPPORTUNITIES
    - Where to reduce spending easily
    - Subscription audit recommendations
    - Alternative cheaper options
    
    🎯 PERSONALIZED RECOMMENDATIONS
    - Specific actions to take this month
    - Apps or tools to help track spending
    - Behavioral changes to consider
    
    Focus on actionable insights that can immediately improve their financial health.
    """,
    
    "investment_advice": """
    Provide investment guidance based on this profile:
    
    Age: {age}
    Income: ${income}
    Current Savings: ${savings}
    Risk Tolerance: {risk_tolerance}
    Investment Goals: {goals}
    Timeline: {timeline}
    
    Please provide comprehensive investment advice including:
    
    🎯 INVESTMENT STRATEGY
    - Recommended asset allocation
    - Investment vehicle suggestions (401k, IRA, taxable accounts)
    - Dollar cost averaging recommendations
    
    📊 PORTFOLIO SUGGESTIONS
    - Low-cost index fund options
    - Diversification strategies
    - International vs domestic allocation
    
    ⚖ RISK MANAGEMENT
    - Risk assessment based on profile
    - Emergency fund requirements before investing
    - Rebalancing frequency recommendations
    
    💰 PRACTICAL STEPS
    - How much to invest monthly
    - Which accounts to prioritize
    - Platform/broker recommendations
    
    ⚠ IMPORTANT DISCLAIMERS
    - This is general guidance, not personalized advice
    - Importance of professional consultation
    - Risk warnings and considerations
    
    Keep advice practical and appropriate for their experience level.
    """,
    
    "debt_payoff": """
    Create a debt payoff strategy for this situation:
    
    DEBT INFORMATION:
    {debt_details}
    
    Monthly Income: ${income}
    Monthly Expenses: ${expenses}
    Available for Debt Payment: ${available_payment}
    
    Please provide a comprehensive debt elimination plan:
    
    📊 DEBT ANALYSIS
    - Total debt amount
    - Weighted average interest rate
    - Current minimum payments
    
    🎯 PAYOFF STRATEGY
    - Debt avalanche vs snowball recommendation
    - Payment prioritization order
    - Timeline to debt freedom
    
    💰 PAYMENT OPTIMIZATION
    - How to allocate extra payments
    - Potential for debt consolidation
    - Balance transfer opportunities
    
    📈 ACCELERATION TACTICS
    - Ways to free up more money for debt
    - Side income suggestions
    - Expense cuts that won't hurt lifestyle
    
    🏆 MOTIVATION & MILESTONES
    - Celebration checkpoints
    - Progress tracking methods
    - Staying motivated during the journey
    
    Provide specific timelines and payment amounts for their situation.
    """,
    
    "emergency_fund": """
    Help create an emergency fund strategy:
    
    Monthly Expenses: ${monthly_expenses}
    Current Savings: ${current_savings}
    Income Stability: {income_stability}
    Family Size: {family_size}
    
    Please provide emergency fund guidance:
    
    🎯 EMERGENCY FUND TARGET
    - Recommended fund size (3-6+ months)
    - Reasoning based on their situation
    - Priority level vs other goals
    
    💰 SAVINGS STRATEGY
    - How much to save monthly
    - Timeline to reach full fund
    - Where to keep the money (high-yield savings, etc.)
    
    🏗 BUILDING APPROACH
    - Start with $1,000 mini-emergency fund
    - Gradual building strategy
    - Automation recommendations
    
    🚨 WHEN TO USE
    - True emergencies vs wants
    - Replenishment strategy after use
    - Avoiding the temptation to use for non-emergencies
    
    📊 ACCOUNT OPTIONS
    - High-yield savings accounts
    - Money market accounts
    - CD laddering for part of fund
    
    Make recommendations specific to their risk profile and financial situation.
    """
}

# Quick response templates for common questions
QUICK_RESPONSES = {
    "greeting": "Hello! I'm your personal finance AI assistant. I can help you with budgeting, savings plans, investment advice, debt management, and general financial planning. What would you like to work on today? 💰",
    
    "budget_help": "I'd be happy to help you create or analyze your budget! Please share your income and expense information, and I'll provide personalized recommendations to optimize your spending and increase your savings. 📊",
    
    "savings_help": "Great choice focusing on savings! Whether you need help with an emergency fund, saving for a specific goal, or just want to improve your savings rate, I can create a personalized plan for you. What are your savings goals? 🎯",
    
    "investment_help": "Investment planning is crucial for long-term wealth building! I can help you understand different investment options, create an asset allocation strategy, and provide guidance on getting started. What's your investment experience level? 📈",
    
    "debt_help": "Let's tackle that debt together! I can help you create a strategic payoff plan, whether you have credit cards, student loans, or other debts. The key is having a solid plan and staying motivated. Share your debt details and I'll help! 💪"
}

# Financial education snippets
EDUCATION_SNIPPETS = {
    "compound_interest": "💡 **Compound Interest Tip**: Starting to save even $100/month at age 25 can grow to over $150,000 by age 65 (assuming 7% annual return). Time is your most powerful wealth-building tool!",
    
    "emergency_fund": "🚨 **Emergency Fund Reminder**: Aim for 3-6 months of expenses in a high-yield savings account. This prevents you from going into debt when unexpected expenses arise.",
    
    "debt_snowball": "❄️ **Debt Snowball Method**: Pay minimums on all debts, then put extra money toward the smallest debt first. The psychological wins help maintain motivation!",
    
    "debt_avalanche": "🏔️ **Debt Avalanche Method**: Pay minimums on all debts, then put extra money toward the highest interest rate debt first. Mathematically optimal for saving money!",
    
    "diversification": "🎯 **Diversification**: Don't put all your eggs in one basket! Spread investments across different asset classes, industries, and geographic regions to reduce risk.",
    
    "dollar_cost_averaging": "📊 **Dollar Cost Averaging**: Invest a fixed amount regularly regardless of market conditions. This reduces the impact of market volatility over time."
}

# Validation and error messages
ERROR_MESSAGES = {
    "insufficient_data": "I need more information to provide accurate advice. Could you please share more details about your financial situation?",
    
    "unrealistic_goals": "Your goals might be a bit ambitious given your current financial situation. Let me help you create a more realistic timeline that you can actually achieve.",
    
    "missing_emergency_fund": "⚠️ I notice you don't have an emergency fund mentioned. This should typically be your first priority before other financial goals.",
    
    "high_debt_ratio": "🚨 Your debt-to-income ratio seems quite high. Let's focus on debt reduction strategies before moving to other goals.",
    
    "low_savings_rate": "Your current savings rate might make it challenging to reach your goals quickly. Let's explore ways to increase your savings capacity."
}

def get_system_prompt(context="general_finance"):
    """Get system prompt for specific financial context"""
    return SYSTEM_PROMPTS.get(context, SYSTEM_PROMPTS["general_finance"])

def get_prompt_template(template_name, **kwargs):
    """Get formatted prompt template with provided variables"""
    if template_name not in PROMPT_TEMPLATES:
        return f"Template '{template_name}' not found."
    
    try:
        return PROMPT_TEMPLATES[template_name].format(**kwargs)
    except KeyError as e:
        return f"Missing required variable: {e}"

def get_quick_response(response_type):
    """Get quick response for common queries"""
    return QUICK_RESPONSES.get(response_type, "I'm here to help with your financial questions! What would you like to know?")

def get_education_snippet(topic):
    """Get educational snippet for financial topic"""
    return EDUCATION_SNIPPETS.get(topic, "")

def get_error_message(error_type):
    """Get appropriate error message"""
    return ERROR_MESSAGES.get(error_type, "I'm having trouble processing that request. Could you please try rephrasing?")