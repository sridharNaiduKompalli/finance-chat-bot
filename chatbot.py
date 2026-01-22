from llm_manager import MultiLLMManager
from finance_analyzer import FinanceAnalyzer
import streamlit as st

class FinanceChatbot:
    def __init__(self):
        self.llm_manager = MultiLLMManager()
        self.analyzer = FinanceAnalyzer(self.llm_manager)
        
        # Initialize conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def get_financial_advice(self, user_input, model_choice="gemini", context="general"):
        """Main method to get financial advice"""
        
        # Add context to the prompt based on the type of query
        if context == "budget":
            system_prompt = "You are a budget planning expert. "
        elif context == "investment":
            system_prompt = "You are an investment advisor. "
        elif context == "savings":
            system_prompt = "You are a savings and financial planning expert. "
        else:
            system_prompt = "You are a personal finance advisor. "
        
        # Include conversation history for context
        history_context = ""
        if st.session_state.conversation_history:
            recent_history = st.session_state.conversation_history[-3:]  # Last 3 exchanges
            history_context = "\nRecent conversation:\n" + "\n".join([
                f"User: {h['user']}\nAssistant: {h['assistant']}" 
                for h in recent_history
            ]) + "\n"
        
        full_prompt = f"{system_prompt}{history_context}\nUser: {user_input}\nAssistant:"
        
        response = self.llm_manager.get_response(model_choice, full_prompt)
        
        # Store in conversation history
        st.session_state.conversation_history.append({
            'user': user_input,
            'assistant': response,
            'model': model_choice
        })
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        st.session_state.conversation_history = []
    
    def export_conversation(self):
        """Export conversation history"""
        if st.session_state.conversation_history:
            conversation_text = ""
            for i, exchange in enumerate(st.session_state.conversation_history, 1):
                conversation_text += f"Exchange {i}:\n"
                conversation_text += f"User: {exchange['user']}\n"
                conversation_text += f"Assistant ({exchange['model']}): {exchange['assistant']}\n\n"
            return conversation_text
        return "No conversation history to export."