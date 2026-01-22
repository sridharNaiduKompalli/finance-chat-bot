import streamlit as st
import pandas as pd
import plotly.express as px
from chatbot import FinanceChatbot
from finance_analyzer import FinanceAnalyzer
from config import STREAMLIT_CONFIG
import io
from tax_calculator import SimpleTaxCalculator
from qa_manager import FinanceQAManager

def main():
    # Page configuration
    st.set_page_config(**STREAMLIT_CONFIG)
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
                # Current CSS block (around line 32), add this to the existing <style> section:
# Find the closing </style> tag and add before it:

.qa-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin-bottom: 1rem;
}
.tag {
    background-color: #e9ecef;
    padding: 0.2rem 0.5rem;
    border-radius: 12px;
    font-size: 0.8rem;
    margin-right: 0.5rem;
}
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 Personal Finance AI Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner('Initializing AI models...'):
            st.session_state.chatbot = FinanceChatbot()
        st.success('AI models loaded successfully!')

    # Add these lines right after:
    if 'qa_manager' not in st.session_state:  # ADD THIS BLOCK
        st.session_state.qa_manager = FinanceQAManager()  # ADD THIS LINE
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "🤖 Choose AI Model:",
            ["gemini", "granite_instruct", "granite_code", "llama2", "falcon"],
            help="Different models have different strengths"
        )
        
        # Feature selection
        feature_mode = st.radio(
            "🎯 Choose Feature:",
            ["💬 Chat", "📊 Budget Analysis", "📈 Spending Insights", "💰 Savings Plan","🧾 Tax Calculator","❓ Q&A Community"]
        )
        
        st.divider()
        
        # Quick actions
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chatbot.clear_history()
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.success("Chat history cleared!")
        
        if st.button("📄 Export Conversation"):
            conversation = st.session_state.chatbot.export_conversation()
            st.download_button(
                "💾 Download Chat History",
                conversation,
                "finance_chat_history.txt",
                "text/plain"
            )
    
    # Main content area
    if feature_mode == "💬 Chat":
        chat_interface(model_choice)
    elif feature_mode == "📊 Budget Analysis":
        budget_analysis_interface(model_choice)
    elif feature_mode == "📈 Spending Insights":
        spending_insights_interface(model_choice)
    elif feature_mode == "💰 Savings Plan":
        savings_plan_interface(model_choice)
    elif feature_mode == "🧾 Tax Calculator":  # NEW: Added this condition
        tax_calculator_interface(model_choice)
    elif feature_mode == "❓ Q&A Community":  # ADD THIS LINE
        qa_interface()  # ADD THIS LINE

def chat_interface(model_choice):
    """Main chat interface"""
    st.subheader(f"💬 Chat with {model_choice.title()}")
    
    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input(f"Ask {model_choice} about your finances..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner(f'{model_choice} is thinking...'):
                response = st.session_state.chatbot.get_financial_advice(prompt, model_choice)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def budget_analysis_interface(model_choice):
    """Budget analysis interface"""
    st.subheader("📊 AI Budget Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Enter Your Expenses")
        
        # File upload option
        uploaded_file = st.file_uploader("📁 Upload CSV file", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded data preview:")
            st.dataframe(df.head())
            expenses_data = df
        else:
            # Manual input option
            st.markdown("**Or enter manually:**")
            sample_data = """Category,Amount
Food,500
Transportation,200
Entertainment,150
Utilities,300
Shopping,250"""
            
            expenses_text = st.text_area(
                "Enter expenses (CSV format):",
                value=sample_data,
                height=200
            )
            
            # Convert text to DataFrame
            try:
                from io import StringIO
                expenses_data = pd.read_csv(StringIO(expenses_text))
            except:
                st.error("Please enter valid CSV format")
                return
    
    with col2:
        st.markdown("### 🤖 AI Analysis")
        
        if st.button("🔍 Analyze Budget", type="primary"):
            with st.spinner(f'{model_choice} is analyzing your budget...'):
                analysis = st.session_state.chatbot.analyzer.generate_budget_summary(
                    expenses_data, model_choice
                )
            
            st.markdown("### 📋 Budget Analysis Results")
            st.markdown(analysis)
            
            # Create visualizations
            if isinstance(expenses_data, pd.DataFrame) and len(expenses_data) > 0:
                fig_pie, fig_bar = st.session_state.chatbot.analyzer.create_spending_visualization(expenses_data)
                if fig_pie and fig_bar:
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.plotly_chart(fig_bar, use_container_width=True)

def spending_insights_interface(model_choice):
    """Spending insights interface"""
    st.subheader("📈 Spending Pattern Analysis")
    
    # Sample transaction data
    sample_transactions = """Date,Description,Amount,Category
2024-01-01,Grocery Store,85.50,Food
2024-01-02,Gas Station,45.00,Transportation
2024-01-03,Restaurant,32.75,Food
2024-01-04,Amazon Purchase,156.99,Shopping
2024-01-05,Electricity Bill,89.25,Utilities"""
    
    transactions_text = st.text_area(
        "📝 Enter your transaction data:",
        value=sample_transactions,
        height=200,
        help="Enter your transactions in CSV format"
    )
    
    if st.button("🔍 Analyze Spending Patterns", type="primary"):
        with st.spinner(f'{model_choice} is analyzing your spending patterns...'):
            insights = st.session_state.chatbot.analyzer.get_spending_insights(
                transactions_text, model_choice
            )
        
        st.markdown("### 💡 Spending Insights")
        st.markdown(insights)

def savings_plan_interface(model_choice):
    """Savings plan interface"""
    st.subheader("💰 Personalized Savings Plan")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Financial Information")
        income = st.number_input("💵 Monthly Income ($)", value=5000, step=100)
        expenses = st.number_input("💸 Monthly Expenses ($)", value=3500, step=100)
        
        st.markdown("### 🎯 Financial Goals")
        goals = st.text_area(
            "What are your financial goals?",
            value="Emergency fund: $10,000\nVacation: $3,000\nNew car: $15,000",
            height=100
        )
    
    with col2:
        if st.button("📋 Create Savings Plan", type="primary"):
            with st.spinner(f'{model_choice} is creating your savings plan...'):
                plan = st.session_state.chatbot.analyzer.generate_savings_plan(
                    income, expenses, goals, model_choice
                )
            
            st.markdown("### 📋 Your Personalized Savings Plan")
            st.markdown(plan)
            
            # Display simple metrics
            disposable_income = income - expenses
            savings_rate = (disposable_income / income) * 100 if income > 0 else 0
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("💰 Monthly Surplus", f"${disposable_income:,.2f}")
            with col4:
                st.metric("📈 Savings Rate", f"{savings_rate:.1f}%")
            with col5:
                emergency_months = (disposable_income * 6) if disposable_income > 0 else 0
                st.metric("🚨 Emergency Fund Goal", f"${emergency_months:,.2f}")
def tax_calculator_interface(model_choice):
    """Tax calculator interface"""
    st.subheader("🧾 Simple Tax Calculator")
    
    # Initialize tax calculator
    if 'tax_calc' not in st.session_state:
        st.session_state.tax_calc = SimpleTaxCalculator()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💰 Income Information")
        
        annual_income = st.number_input(
            "Annual Gross Income ($)", 
            value=50000, 
            step=1000,
            min_value=0
        )
        
        filing_status = st.selectbox(
            "Filing Status",
            ["single", "married_joint", "married_separate", "head_of_household"],
            help="Choose your tax filing status"
        )
        
        st.markdown("### 🏛️ Tax Settings")
        
        use_standard_deduction = st.checkbox("Use Standard Deduction", value=True)
        
        if not use_standard_deduction:
            custom_deductions = st.number_input(
                "Custom Deductions ($)", 
                value=15000, 
                step=500,
                min_value=0
            )
        else:
            custom_deductions = None
        
        state_tax_rate = st.slider(
            "State Tax Rate (%)", 
            min_value=0.0, 
            max_value=15.0, 
            value=5.0, 
            step=0.1
        ) / 100
        
        calculate_button = st.button("🧮 Calculate Taxes", type="primary")
    
    with col2:
        if calculate_button:
            with st.spinner('Calculating your taxes...'):
                # Calculate taxes
                tax_result = st.session_state.tax_calc.calculate_total_taxes(
                    annual_income, 
                    filing_status, 
                    state_tax_rate, 
                    custom_deductions
                )
                
                st.markdown("### 📊 Tax Calculation Results")
                
                # Display key metrics
                col3, col4 = st.columns(2)
                
                with col3:
                    st.metric("💵 Gross Income", f"${tax_result['gross_income']:,.2f}")
                    st.metric("🏛️ Federal Tax", f"${tax_result['federal_tax']:,.2f}")
                    st.metric("🏠 State Tax", f"${tax_result['state_tax']:,.2f}")
                
                with col4:
                    st.metric("👥 Payroll Taxes", f"${tax_result['payroll_taxes']:,.2f}")
                    st.metric("💸 Total Taxes", f"${tax_result['total_tax']:,.2f}")
                    st.metric("💰 Net Income", f"${tax_result['net_income']:,.2f}")
                
                # Tax rate information
                st.markdown("### 📈 Tax Rate Analysis")
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.metric("📊 Total Tax Rate", f"{tax_result['total_tax_rate']:.2f}%")
                
                with col6:
                    effective_rate = tax_result['federal_details']['effective_rate']
                    st.metric("🎯 Federal Effective Rate", f"{effective_rate:.2f}%")
                
                with col7:
                    marginal_rate = tax_result['federal_details']['marginal_rate']
                    st.metric("📈 Federal Marginal Rate", f"{marginal_rate:.1f}%")
                
                # Detailed breakdown
                with st.expander("🔍 Detailed Tax Breakdown"):
                    st.markdown("**Federal Tax Details:**")
                    fed_details = tax_result['federal_details']
                    st.write(f"• Taxable Income: ${fed_details['taxable_income']:,.2f}")
                    st.write(f"• Deductions Used: ${fed_details['deductions']:,.2f}")
                    
                    st.markdown("**Payroll Tax Details:**")
                    payroll_details = tax_result['payroll_details']
                    st.write(f"• Social Security Tax: ${payroll_details['social_security_tax']:,.2f}")
                    st.write(f"• Medicare Tax: ${payroll_details['medicare_tax']:,.2f}")
                
                # AI Analysis (optional)
                if st.button("🤖 Get AI Tax Analysis"):
                    with st.spinner(f'{model_choice} is analyzing your tax situation...'):
                        analysis_prompt = f"""
                        Analyze this tax calculation and provide insights:
                        
                        Gross Income: ${tax_result['gross_income']:,.2f}
                        Total Tax: ${tax_result['total_tax']:,.2f}
                        Net Income: ${tax_result['net_income']:,.2f}
                        Total Tax Rate: {tax_result['total_tax_rate']:.2f}%
                        Federal Effective Rate: {tax_result['federal_details']['effective_rate']:.2f}%
                        
                        Provide:
                        1. 💡 Tax optimization suggestions
                        2. 📊 How this compares to average tax rates
                        3. 🎯 Strategies to reduce tax burden
                        4. 💰 Retirement/investment tax benefits to consider
                        """
                        
                        # Use your existing chatbot for analysis
                        analysis = st.session_state.chatbot.get_financial_advice(
                            analysis_prompt, model_choice, context="taxes"
                        )
                        
                        st.markdown("### 🤖 AI Tax Analysis")
                        st.markdown(analysis)
def qa_interface():
    """Q&A Community interface"""
    st.subheader("❓ Finance Q&A Community")
    
    # Tabs for different Q&A actions
    tab1, tab2, tab3 = st.tabs(["🔍 Browse & Search", "❓ Ask Question", "💡 Answer Questions"])
    
    with tab1:
        st.markdown("### 🔍 Search Finance Questions")
        
        # Search filters
        col1, col2 = st.columns([2, 1])
        with col1:
            search_keyword = st.text_input("🔎 Search by keyword:", placeholder="e.g., investment, tax, savings")
        with col2:
            categories = ["all"] + st.session_state.qa_manager.get_categories()
            category_filter = st.selectbox("📂 Category:", categories)
        
        # Popular tags
        popular_tags = st.session_state.qa_manager.get_popular_tags()
        if popular_tags:
            st.markdown("**🏷️ Popular Tags:**")
            tag_cols = st.columns(len(popular_tags))
            for i, tag in enumerate(popular_tags):
                with tag_cols[i]:
                    if st.button(f"#{tag}", key=f"tag_{tag}"):
                        search_keyword = tag
                        st.rerun()
        
        # Search results
        results = st.session_state.qa_manager.search_questions(search_keyword, category_filter)
        
        st.markdown(f"### 📋 Results ({len(results)} found)")
        
        if results:
            for qa in results:
                with st.container():
                    st.markdown(f"""
                    <div class="qa-card">
                        <h4>❓ {qa['question']}</h4>
                        <p><strong>Category:</strong> {qa['category'].title()} | 
                           <strong>Author:</strong> {qa['author']} | 
                           <strong>Votes:</strong> {qa['votes']} 👍</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show answer if available
                    if qa['answer']:
                        st.markdown(f"**✅ Answer:** {qa['answer']}")
                        if 'answer_author' in qa:
                            st.caption(f"Answered by: {qa['answer_author']}")
                    else:
                        st.markdown("*⏳ No answers yet - be the first to answer!*")
                    
                    # Show tags
                    if qa['tags']:
                        tag_html = " ".join([f'<span class="tag">#{tag}</span>' for tag in qa['tags']])
                        st.markdown(f"**Tags:** {tag_html}", unsafe_allow_html=True)
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("👍 Upvote", key=f"upvote_{qa['id']}"):
                            st.session_state.qa_manager.upvote_question(qa['id'])
                            st.rerun()
                    
                    with col2:
                        if not qa['answer'] and st.button("💡 Answer", key=f"answer_{qa['id']}"):
                            st.session_state.current_answering = qa['id']
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No questions found. Try different keywords or ask a new question!")
    
    with tab2:
        st.markdown("### ❓ Ask a New Question")
        
        with st.form("ask_question_form"):
            question = st.text_area("💭 Your question:", 
                                   placeholder="e.g., What's the best way to invest $10,000?",
                                   height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("📂 Category:", 
                                      ["investment", "tax", "savings", "budgeting", "insurance", "retirement"])
            with col2:
                author = st.text_input("👤 Your name:", value="Anonymous")
            
            tags_input = st.text_input("🏷️ Tags (comma separated):", 
                                     placeholder="e.g., stocks, beginner, long-term")
            
            submitted = st.form_submit_button("🚀 Ask Question", type="primary")
            
            if submitted and question:
                tags = [tag.strip().lower() for tag in tags_input.split(",") if tag.strip()]
                question_id = st.session_state.qa_manager.add_question(question, category, author, tags)
                st.success(f"✅ Question added successfully! (ID: {question_id})")
                st.rerun()
    
    with tab3:
        st.markdown("### 💡 Answer Questions")
        
        # Show unanswered questions
        unanswered = [qa for qa in st.session_state.qa_manager.search_questions() if not qa['answer']]
        
        if unanswered:
            st.markdown(f"**📝 {len(unanswered)} questions waiting for answers:**")
            
            for qa in unanswered[:5]:  # Show first 5 unanswered
                with st.expander(f"❓ {qa['question'][:80]}..."):
                    st.markdown(f"**Full Question:** {qa['question']}")
                    st.markdown(f"**Category:** {qa['category'].title()}")
                    st.markdown(f"**Asked by:** {qa['author']}")
                    
                    # Answer form
                    with st.form(f"answer_form_{qa['id']}"):
                        answer = st.text_area("Your answer:", 
                                            placeholder="Share your knowledge...",
                                            height=100,
                                            key=f"answer_text_{qa['id']}")
                        answer_author = st.text_input("Your name:", value="Anonymous", 
                                                    key=f"answer_author_{qa['id']}")
                        
                        if st.form_submit_button("✅ Submit Answer"):
                            if answer:
                                st.session_state.qa_manager.add_answer(qa['id'], answer, answer_author)
                                st.success("Answer submitted successfully!")
                                st.rerun()
        else:
            st.info("🎉 All questions have been answered! Check back later for new questions.")

if __name__ == "__main__":
    main()

