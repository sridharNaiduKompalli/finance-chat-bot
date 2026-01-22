"""
Unit tests for finance features and analyzer functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from finance_analyzer import FinanceAnalyzer
from chatbot import FinanceChatbot
from llm_manager import MultiLLMManager


class TestFinanceAnalyzer(unittest.TestCase):
    """Test cases for the FinanceAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.mock_llm_manager = Mock(spec=MultiLLMManager)
        self.analyzer = FinanceAnalyzer(self.mock_llm_manager)
        
        # Sample test data
        self.sample_expenses_dict = {
            'Food': 500,
            'Transportation': 200,
            'Entertainment': 150,
            'Utilities': 300
        }
        
        self.sample_expenses_df = pd.DataFrame([
            {'Category': 'Food', 'Amount': 500},
            {'Category': 'Transportation', 'Amount': 200},
            {'Category': 'Entertainment', 'Amount': 150},
            {'Category': 'Utilities', 'Amount': 300}
        ])
        
        self.sample_transactions = """Date,Description,Amount,Category
2024-01-01,Grocery Store,85.50,Food
2024-01-02,Gas Station,45.00,Transportation
2024-01-03,Restaurant,32.75,Food
2024-01-04,Amazon Purchase,156.99,Shopping"""
    
    def test_initialization(self):
        """Test that FinanceAnalyzer initializes correctly"""
        self.assertIsInstance(self.analyzer, FinanceAnalyzer)
        self.assertEqual(self.analyzer.llm_manager, self.mock_llm_manager)
    
    def test_generate_budget_summary_string_input(self):
        """Test budget summary generation with string input"""
        self.mock_llm_manager.get_response.return_value = "Budget analysis result"
        
        result = self.analyzer.generate_budget_summary("Sample expense data", "gemini")
        
        self.mock_llm_manager.get_response.assert_called_once()
        self.assertEqual(result, "Budget analysis result")
        
        # Check that the prompt contains expected elements
        call_args = self.mock_llm_manager.get_response.call_args
        self.assertEqual(call_args[0][0], "gemini")  # model choice
        prompt = call_args[0][1]
        self.assertIn("Sample expense data", prompt)
        self.assertIn("TOTAL SPENDING ANALYSIS", prompt)
    
    def test_generate_budget_summary_dataframe_input(self):
        """Test budget summary generation with DataFrame input"""
        self.mock_llm_manager.get_response.return_value = "DataFrame budget analysis"
        
        result = self.analyzer.generate_budget_summary(self.sample_expenses_df, "granite_instruct")
        
        self.mock_llm_manager.get_response.assert_called_once()
        self.assertEqual(result, "DataFrame budget analysis")
        
        # Check that DataFrame was converted to string
        call_args = self.mock_llm_manager.get_response.call_args
        prompt = call_args[0][1]
        self.assertIn("Food", prompt)
        self.assertIn("500", prompt)
    
    def test_get_spending_insights(self):
        """Test spending insights generation"""
        self.mock_llm_manager.get_response.return_value = "Spending insights result"
        
        result = self.analyzer.get_spending_insights(self.sample_transactions, "granite_instruct")
        
        self.mock_llm_manager.get_response.assert_called_once_with("granite_instruct", unittest.mock.ANY)
        self.assertEqual(result, "Spending insights result")
        
        # Check prompt content
        call_args = self.mock_llm_manager.get_response.call_args
        prompt = call_args[0][1]
        self.assertIn("SPENDING PATTERNS", prompt)
        self.assertIn("TOP SPENDING CATEGORIES", prompt)
        self.assertIn(self.sample_transactions, prompt)
    
    def test_create_spending_visualization_dict_input(self):
        """Test visualization creation with dictionary input"""
        fig_pie, fig_bar = self.analyzer.create_spending_visualization(self.sample_expenses_dict)
        
        self.assertIsNotNone(fig_pie)
        self.assertIsNotNone(fig_bar)
        
        # Check that the figures have the expected properties
        self.assertIn("Spending by Category", fig_pie.layout.title.text)
        self.assertIn("Category-wise Spending", fig_bar.layout.title.text)
    
    def test_create_spending_visualization_dataframe_input(self):
        """Test visualization creation with DataFrame input"""
        fig_pie, fig_bar = self.analyzer.create_spending_visualization(self.sample_expenses_df)
        
        self.assertIsNotNone(fig_pie)
        self.assertIsNotNone(fig_bar)
    
    def test_create_spending_visualization_invalid_input(self):
        """Test visualization creation with invalid input"""
        result = self.analyzer.create_spending_visualization("invalid input")
        
        self.assertIsNone(result)
    
    def test_generate_savings_plan(self):
        """Test savings plan generation"""
        self.mock_llm_manager.get_response.return_value = "Savings plan result"
        
        result = self.analyzer.generate_savings_plan(5000, 3500, "Emergency fund, Vacation", "gemini")
        
        self.mock_llm_manager.get_response.assert_called_once_with("gemini", unittest.mock.ANY)
        self.assertEqual(result, "Savings plan result")
        
        # Check prompt content
        call_args = self.mock_llm_manager.get_response.call_args
        prompt = call_args[0][1]
        self.assertIn("5000", prompt)
        self.assertIn("3500", prompt)
        self.assertIn("Emergency fund, Vacation", prompt)
        self.assertIn("SAVINGS CAPACITY ANALYSIS", prompt)


class TestFinanceChatbot(unittest.TestCase):
    """Test cases for the FinanceChatbot class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock streamlit session state
        self.mock_session_state = MagicMock()
        self.mock_session_state.conversation_history = []
        
        with patch('streamlit.session_state', self.mock_session_state):
            self.chatbot = FinanceChatbot()
    
    def test_initialization(self):
        """Test that FinanceChatbot initializes correctly"""
        self.assertIsInstance(self.chatbot, FinanceChatbot)
        self.assertIsInstance(self.chatbot.llm_manager, MultiLLMManager)
        self.assertIsInstance(self.chatbot.analyzer, FinanceAnalyzer)
    
    @patch('streamlit.session_state')
    def test_get_financial_advice_general(self, mock_session_state):
        """Test getting general financial advice"""
        mock_session_state.conversation_history = []
        
        with patch.object(self.chatbot.llm_manager, 'get_response', return_value="Financial advice response"):
            result = self.chatbot.get_financial_advice("How can I save money?", "gemini", "general")
            
            self.assertEqual(result, "Financial advice response")
            self.chatbot.llm_manager.get_response.assert_called_once()
    
    @patch('streamlit.session_state')
    def test_get_financial_advice_with_context(self, mock_session_state):
        """Test getting financial advice with specific context"""
        mock_session_state.conversation_history = []
        
        with patch.object(self.chatbot.llm_manager, 'get_response', return_value="Budget advice"):
            result = self.chatbot.get_financial_advice("Help me budget", "gemini", "budget")
            
            call_args = self.chatbot.llm_manager.get_response.call_args
            prompt = call_args[0][1]
            self.assertIn("budget planning expert", prompt)
    
    @patch('streamlit.session_state')
    def test_get_financial_advice_with_history(self, mock_session_state):
        """Test getting advice with conversation history"""
        mock_session_state.conversation_history = [
            {'user': 'Previous question', 'assistant': 'Previous answer', 'model': 'gemini'}
        ]
        
        with patch.object(self.chatbot.llm_manager, 'get_response', return_value="Response with context"):
            result = self.chatbot.get_financial_advice("Follow up question", "gemini")
            
            call_args = self.chatbot.llm_manager.get_response.call_args
            prompt = call_args[0][1]
            self.assertIn("Recent conversation", prompt)
            self.assertIn("Previous question", prompt)
    
    @patch('streamlit.session_state')
    def test_conversation_history_storage(self, mock_session_state):
        """Test that conversation history is properly stored"""
        mock_session_state.conversation_history = []
        
        with patch.object(self.chatbot.llm_manager, 'get_response', return_value="Test response"):
            self.chatbot.get_financial_advice("Test question", "gemini")
            
            # Check that conversation was stored
            mock_session_state.conversation_history.append.assert_called_once()
            stored_conversation = mock_session_state.conversation_history.append.call_args[0][0]
            self.assertEqual(stored_conversation['user'], "Test question")
            self.assertEqual(stored_conversation['assistant'], "Test response")
            self.assertEqual(stored_conversation['model'], "gemini")
    
    @patch('streamlit.session_state')
    def test_clear_history(self, mock_session_state):
        """Test clearing conversation history"""
        mock_session_state.conversation_history = ['some', 'history']
        
        self.chatbot.clear_history()
        
        # Verify history was cleared
        self.assertEqual(mock_session_state.conversation_history, [])
    
    @patch('streamlit.session_state')
    def test_export_conversation_with_history(self, mock_session_state):
        """Test exporting conversation with existing history"""
        mock_session_state.conversation_history = [
            {'user': 'Question 1', 'assistant': 'Answer 1', 'model': 'gemini'},
            {'user': 'Question 2', 'assistant': 'Answer 2', 'model': 'granite_instruct'}
        ]
        
        result = self.chatbot.export_conversation()
        
        self.assertIn("Exchange 1:", result)
        self.assertIn("Question 1", result)
        self.assertIn("Answer 1", result)
        self.assertIn("gemini", result)
        self.assertIn("Exchange 2:", result)
        self.assertIn("Question 2", result)
        self.assertIn("Answer 2", result)
        self.assertIn("granite_instruct", result)
    
    @patch('streamlit.session_state')
    def test_export_conversation_empty_history(self, mock_session_state):
        """Test exporting conversation with empty history"""
        mock_session_state.conversation_history = []
        
        result = self.chatbot.export_conversation()
        
        self.assertEqual(result, "No conversation history to export.")


class TestFinanceAnalyzerVisualization(unittest.TestCase):
    """Test cases specifically for visualization functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm_manager = Mock()
        self.analyzer = FinanceAnalyzer(self.mock_llm_manager)
    
    def test_visualization_data_processing(self):
        """Test that visualization correctly processes different data formats"""
        # Test with dictionary
        data_dict = {'Food': 500, 'Transport': 300, 'Entertainment': 200}
        fig_pie, fig_bar = self.analyzer.create_spending_visualization(data_dict)
        
        self.assertIsNotNone(fig_pie)
        self.assertIsNotNone(fig_bar)
        
        # Test with DataFrame
        data_df = pd.DataFrame([
            {'Category': 'Food', 'Amount': 500},
            {'Category': 'Transport', 'Amount': 300},
            {'Category': 'Entertainment', 'Amount': 200}
        ])
        fig_pie, fig_bar = self.analyzer.create_spending_visualization(data_df)
        
        self.assertIsNotNone(fig_pie)
        self.assertIsNotNone(fig_bar)
    
    def test_visualization_chart_properties(self):
        """Test that charts have correct properties"""
        data = {'Food': 500, 'Transport': 300}
        fig_pie, fig_bar = self.analyzer.create_spending_visualization(data)
        
        # Check pie chart
        self.assertIn("💰 Spending by Category", fig_pie.layout.title.text)
        
        # Check bar chart
        self.assertIn("📊 Category-wise Spending", fig_bar.layout.title.text)
        
        # Verify data is present in charts
        pie_data = fig_pie.data[0]
        self.assertIn('Food', pie_data.labels)
        self.assertIn('Transport', pie_data.labels)
        self.assertIn(500, pie_data.values)
        self.assertIn(300, pie_data.values)


class TestFinancePromptTemplates(unittest.TestCase):
    """Test cases for finance prompt templates"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import the prompt templates
        try:
            from data.templates.finance_prompts import (
                get_system_prompt, get_prompt_template, 
                get_quick_response, get_education_snippet
            )
            self.get_system_prompt = get_system_prompt
            self.get_prompt_template = get_prompt_template
            self.get_quick_response = get_quick_response
            self.get_education_snippet = get_education_snippet
        except ImportError:
            self.skipTest("Finance prompts module not available")
    
    def test_system_prompt_retrieval(self):
        """Test system prompt retrieval"""
        general_prompt = self.get_system_prompt("general_finance")
        self.assertIn("personal finance advisor", general_prompt.lower())
        
        budget_prompt = self.get_system_prompt("budget_planning")
        self.assertIn("budget planning", budget_prompt.lower())
        
        # Test default fallback
        default_prompt = self.get_system_prompt("nonexistent_context")
        self.assertEqual(default_prompt, self.get_system_prompt("general_finance"))
    
    def test_prompt_template_formatting(self):
        """Test prompt template formatting"""
        template = self.get_prompt_template(
            "budget_analysis", 
            data="Food: $500, Transport: $300"
        )
        
        self.assertIn("Food: $500", template)
        self.assertIn("Transport: $300", template)
        self.assertIn("SPENDING OVERVIEW", template)
    
    def test_prompt_template_missing_variables(self):
        """Test prompt template with missing variables"""
        template = self.get_prompt_template("budget_analysis")
        self.assertIn("Missing required variable", template)
    
    def test_quick_responses(self):
        """Test quick response retrieval"""
        greeting = self.get_quick_response("greeting")
        self.assertIn("personal finance AI assistant", greeting.lower())
        
        budget_help = self.get_quick_response("budget_help")
        self.assertIn("budget", budget_help.lower())
        
        # Test default response
        default_response = self.get_quick_response("nonexistent_type")
        self.assertIn("financial questions", default_response.lower())
    
    def test_education_snippets(self):
        """Test education snippet retrieval"""
        compound_snippet = self.get_education_snippet("compound_interest")
        self.assertIn("Compound Interest", compound_snippet)
        
        emergency_snippet = self.get_education_snippet("emergency_fund")
        self.assertIn("Emergency Fund", emergency_snippet)
        
        # Test empty return for nonexistent topic
        empty_snippet = self.get_education_snippet("nonexistent_topic")
        self.assertEqual(empty_snippet, "")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete finance scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm_manager = Mock()
        self.analyzer = FinanceAnalyzer(self.mock_llm_manager)
    
    def test_complete_budget_analysis_workflow(self):
        """Test complete budget analysis workflow"""
        # Mock LLM response
        self.mock_llm_manager.get_response.return_value = "Complete budget analysis with recommendations"
        
        # Test data
        expenses_data = pd.DataFrame([
            {'Category': 'Food', 'Amount': 600},
            {'Category': 'Housing', 'Amount': 1200},
            {'Category': 'Transportation', 'Amount': 400},
            {'Category': 'Entertainment', 'Amount': 200}
        ])
        
        # Generate analysis
        analysis = self.analyzer.generate_budget_summary(expenses_data, "gemini")
        
        # Generate visualizations
        fig_pie, fig_bar = self.analyzer.create_spending_visualization(expenses_data)
        
        # Verify results
        self.assertIsNotNone(analysis)
        self.assertIsNotNone(fig_pie)
        self.assertIsNotNone(fig_bar)
        self.mock_llm_manager.get_response.assert_called_once()
    
    def test_savings_plan_workflow(self):
        """Test complete savings plan generation workflow"""
        self.mock_llm_manager.get_response.return_value = "Detailed savings plan with specific recommendations"
        
        # Test parameters
        income = 5000
        expenses = 3500
        goals = "Emergency fund: $10,000, Vacation: $3,000, Car: $15,000"
        
        # Generate savings plan
        plan = self.analyzer.generate_savings_plan(income, expenses, goals, "gemini")
        
        # Verify call was made with correct parameters
        self.mock_llm_manager.get_response.assert_called_once()
        call_args = self.mock_llm_manager.get_response.call_args
        prompt = call_args[0][1]
        
        self.assertIn(str(income), prompt)
        self.assertIn(str(expenses), prompt)
        self.assertIn(goals, prompt)
        self.assertIn("SAVINGS CAPACITY", prompt)


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFinanceAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestFinanceChatbot))
    suite.addTests(loader.loadTestsFromTestCase(TestFinanceAnalyzerVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestFinancePromptTemplates))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        # Print failures and errors
        for test, traceback in result.failures:
            print(f"\nFAILED: {test}")
            print(traceback)
        
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(traceback)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)