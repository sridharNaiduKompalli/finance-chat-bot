"""
Unit tests for LLM models and management functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_manager import MultiLLMManager
import config


class TestMultiLLMManager(unittest.TestCase):
    """Test cases for the MultiLLMManager class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.manager = MultiLLMManager()
    
    def test_initialization(self):
        """Test that MultiLLMManager initializes correctly"""
        self.assertIsInstance(self.manager, MultiLLMManager)
        self.assertIsNotNone(self.manager.models)
        self.assertIn('granite_code', self.manager.models)
        self.assertIn('granite_instruct', self.manager.models)
    
    @patch('llm_manager.InferenceClient')
    def test_huggingface_client_initialization(self, mock_client):
        """Test HuggingFace client initialization"""
        manager = MultiLLMManager()
        mock_client.assert_called_once()
    
    @patch('llm_manager.genai')
    @patch('llm_manager.GOOGLE_API_KEY', 'test_key')
    def test_gemini_initialization_with_key(self, mock_genai):
        """Test Gemini initialization when API key is provided"""
        manager = MultiLLMManager()
        mock_genai.configure.assert_called_once_with(api_key='test_key')
        mock_genai.GenerativeModel.assert_called_once_with('gemini-pro')
    
    @patch('llm_manager.GOOGLE_API_KEY', None)
    def test_gemini_initialization_without_key(self):
        """Test Gemini initialization when API key is not provided"""
        manager = MultiLLMManager()
        self.assertIsNone(manager.gemini_model)
    
    def test_model_list_completeness(self):
        """Test that all expected models are available"""
        expected_models = ['granite_code', 'granite_instruct', 'llama2', 'falcon']
        for model in expected_models:
            self.assertIn(model, self.manager.models)
    
    @patch.object(MultiLLMManager, 'query_huggingface_model')
    def test_get_response_huggingface_model(self, mock_query):
        """Test getting response from HuggingFace model"""
        mock_query.return_value = "Test response"
        
        response = self.manager.get_response('granite_instruct', 'Test prompt')
        
        mock_query.assert_called_once_with('granite_instruct', 'Test prompt')
        self.assertEqual(response, "Test response")
    
    @patch.object(MultiLLMManager, 'query_gemini')
    def test_get_response_gemini_model(self, mock_query):
        """Test getting response from Gemini model"""
        mock_query.return_value = "Gemini response"
        
        response = self.manager.get_response('gemini', 'Test prompt')
        
        mock_query.assert_called_once_with('Test prompt')
        self.assertEqual(response, "Gemini response")
    
    def test_query_huggingface_model_invalid_model(self):
        """Test querying invalid HuggingFace model"""
        response = self.manager.query_huggingface_model('invalid_model', 'Test prompt')
        self.assertEqual(response, "Model not found")
    
    def test_query_gemini_without_api_key(self):
        """Test querying Gemini without API key"""
        manager = MultiLLMManager()
        manager.gemini_model = None
        
        response = manager.query_gemini('Test prompt')
        self.assertEqual(response, "Gemini API key not configured")
    
    @patch('llm_manager.InferenceClient')
    def test_query_huggingface_model_success(self, mock_client_class):
        """Test successful HuggingFace model query"""
        # Setup mock
        mock_client = Mock()
        mock_client.text_generation.return_value = "Generated response"
        mock_client_class.return_value = mock_client
        
        manager = MultiLLMManager()
        
        response = manager.query_huggingface_model('granite_instruct', 'Test prompt')
        
        # Verify the client was called with correct parameters
        mock_client.text_generation.assert_called_once_with(
            prompt='Test prompt',
            model=config.GRANITE_MODELS['granite_instruct'],
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
        self.assertEqual(response, "Generated response")
    
    @patch('llm_manager.InferenceClient')
    def test_query_huggingface_model_exception(self, mock_client_class):
        """Test HuggingFace model query with exception"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.text_generation.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        manager = MultiLLMManager()
        
        response = manager.query_huggingface_model('granite_instruct', 'Test prompt')
        
        self.assertIn("Error with granite_instruct", response)
        self.assertIn("API Error", response)
    
    def test_query_gemini_success(self):
        """Test successful Gemini query"""
        # Setup mock Gemini model
        mock_response = Mock()
        mock_response.text = "Gemini generated response"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        
        manager = MultiLLMManager()
        manager.gemini_model = mock_model
        
        response = manager.query_gemini('Test prompt')
        
        mock_model.generate_content.assert_called_once_with('Test prompt')
        self.assertEqual(response, "Gemini generated response")
    
    def test_query_gemini_exception(self):
        """Test Gemini query with exception"""
        # Setup mock Gemini model to raise exception
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Gemini API Error")
        
        manager = MultiLLMManager()
        manager.gemini_model = mock_model
        
        response = manager.query_gemini('Test prompt')
        
        self.assertIn("Gemini Error", response)
        self.assertIn("Gemini API Error", response)


class TestConfigValidation(unittest.TestCase):
    """Test cases for configuration validation"""
    
    def test_granite_models_config(self):
        """Test that Granite models are properly configured"""
        self.assertIn('granite_code', config.GRANITE_MODELS)
        self.assertIn('granite_instruct', config.GRANITE_MODELS)
        self.assertIsInstance(config.GRANITE_MODELS['granite_code'], str)
        self.assertIsInstance(config.GRANITE_MODELS['granite_instruct'], str)
    
    def test_other_models_config(self):
        """Test that other models are properly configured"""
        self.assertIn('llama2', config.OTHER_MODELS)
        self.assertIn('falcon', config.OTHER_MODELS)
        self.assertIsInstance(config.OTHER_MODELS['llama2'], str)
        self.assertIsInstance(config.OTHER_MODELS['falcon'], str)
    
    def test_streamlit_config(self):
        """Test that Streamlit configuration is valid"""
        self.assertIn('page_title', config.STREAMLIT_CONFIG)
        self.assertIn('page_icon', config.STREAMLIT_CONFIG)
        self.assertIn('layout', config.STREAMLIT_CONFIG)
        self.assertEqual(config.STREAMLIT_CONFIG['layout'], 'wide')


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = MultiLLMManager()
        self.test_prompts = [
            "What is a budget?",
            "How can I save money?",
            "What are good investment options?",
            "How do I pay off debt?"
        ]
    
    def test_model_response_format(self):
        """Test that model responses are strings"""
        with patch.object(self.manager, 'query_huggingface_model', return_value="Test response"):
            response = self.manager.get_response('granite_instruct', 'Test prompt')
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
    
    def test_all_models_accessibility(self):
        """Test that all configured models can be accessed"""
        all_models = list(self.manager.models.keys()) + ['gemini']
        
        for model in all_models:
            with self.subTest(model=model):
                # Mock the actual API calls to avoid real API usage in tests
                if model == 'gemini':
                    with patch.object(self.manager, 'query_gemini', return_value=f"Response from {model}"):
                        response = self.manager.get_response(model, 'Test prompt')
                else:
                    with patch.object(self.manager, 'query_huggingface_model', return_value=f"Response from {model}"):
                        response = self.manager.get_response(model, 'Test prompt')
                
                self.assertIsInstance(response, str)
                self.assertIn(model, response.lower())


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMultiLLMManager))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)