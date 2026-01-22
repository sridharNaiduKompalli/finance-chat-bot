# models/model_configs.py
"""
Configuration settings for different AI models used in the finance chatbot
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """Enumeration of supported model types"""
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"
    OPENAI = "openai"
    LOCAL = "local"

class ModelCapability(Enum):
    """Model capabilities for different tasks"""
    CHAT = "chat"
    CODE = "code"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"

@dataclass
class ModelConfig:
    """Configuration class for individual models"""
    name: str
    model_id: str
    model_type: ModelType
    capabilities: List[ModelCapability]
    max_tokens: int
    temperature: float
    description: str
    api_endpoint: Optional[str] = None
    requires_auth: bool = True
    cost_per_token: Optional[float] = None
    context_window: int = 4096
    supports_streaming: bool = False
    prompt_template: Optional[str] = None

# Granite Models (IBM)
GRANITE_MODELS = {
    "granite_code": ModelConfig(
        name="Granite Code",
        model_id="ibm-granite/granite-3b-code-base",
        model_type=ModelType.HUGGINGFACE,
        capabilities=[ModelCapability.CODE, ModelCapability.ANALYSIS],
        max_tokens=2048,
        temperature=0.3,
        description="IBM Granite model optimized for code generation and analysis. Great for financial calculations and data processing.",
        context_window=4096,
        supports_streaming=False,
        prompt_template="### Code:\n{prompt}\n### Response:"
    ),
    
    "granite_instruct": ModelConfig(
        name="Granite Instruct",
        model_id="ibm-granite/granite-7b-instruct",
        model_type=ModelType.HUGGINGFACE,
        capabilities=[ModelCapability.CHAT, ModelCapability.ANALYSIS, ModelCapability.SUMMARIZATION],
        max_tokens=1024,
        temperature=0.7,
        description="IBM Granite instruction-tuned model for general financial advice and conversation.",
        context_window=4096,
        supports_streaming=False,
        prompt_template="Human: {prompt}\nAssistant:"
    )
}

# Meta LLaMA Models
LLAMA_MODELS = {
    "llama2": ModelConfig(
        name="LLaMA 2 Chat",
        model_id="meta-llama/Llama-2-7b-chat-hf",
        model_type=ModelType.HUGGINGFACE,
        capabilities=[ModelCapability.CHAT, ModelCapability.ANALYSIS],
        max_tokens=1024,
        temperature=0.7,
        description="Meta's LLaMA 2 model fine-tuned for conversational AI. Excellent for financial discussions.",
        context_window=4096,
        supports_streaming=True,
        prompt_template="<s>[INST] {prompt} [/INST]"
    ),
    
    "llama2_13b": ModelConfig(
        name="LLaMA 2 13B Chat",
        model_id="meta-llama/Llama-2-13b-chat-hf",
        model_type=ModelType.HUGGINGFACE,
        capabilities=[ModelCapability.CHAT, ModelCapability.ANALYSIS, ModelCapability.SUMMARIZATION],
        max_tokens=2048,
        temperature=0.7,
        description="Larger LLaMA 2 model with enhanced reasoning capabilities for complex financial analysis.",
        context_window=4096,
        supports_streaming=True,
        prompt_template="<s>[INST] {prompt} [/INST]"
    )
}

# Falcon Models
FALCON_MODELS = {
    "falcon": ModelConfig(
        name="Falcon Instruct",
        model_id="tiiuae/falcon-7b-instruct",
        model_type=ModelType.HUGGINGFACE,
        capabilities=[ModelCapability.CHAT, ModelCapability.ANALYSIS],
        max_tokens=1024,
        temperature=0.7,
        description="TII's Falcon model trained on diverse data. Good for general financial advice.",
        context_window=2048,
        supports_streaming=False,
        prompt_template="User: {prompt}\nAssistant:"
    ),
    
    "falcon_40b": ModelConfig(
        name="Falcon 40B Instruct",
        model_id="tiiuae/falcon-40b-instruct",
        model_type=ModelType.HUGGINGFACE,
        capabilities=[ModelCapability.CHAT, ModelCapability.ANALYSIS, ModelCapability.CODE],
        max_tokens=2048,
        temperature=0.7,
        description="Large Falcon model with superior reasoning for complex financial scenarios.",
        context_window=2048,
        supports_streaming=False,
        prompt_template="User: {prompt}\nAssistant:"
    )
}

# Google Models
GOOGLE_MODELS = {
    "gemini": ModelConfig(
        name="Gemini Pro",
        model_id="gemini-pro",
        model_type=ModelType.GOOGLE,
        capabilities=[ModelCapability.CHAT, ModelCapability.ANALYSIS, ModelCapability.SUMMARIZATION],
        max_tokens=2048,
        temperature=0.7,
        description="Google's Gemini Pro model with multimodal capabilities. Excellent for comprehensive financial analysis.",
        context_window=30720,
        supports_streaming=True,
        cost_per_token=0.00025,
        prompt_template="{prompt}"
    ),
    
    "gemini_pro_vision": ModelConfig(
        name="Gemini Pro Vision",
        model_id="gemini-pro-vision",
        model_type=ModelType.GOOGLE,
        capabilities=[ModelCapability.CHAT, ModelCapability.ANALYSIS],
        max_tokens=2048,
        temperature=0.7,
        description="Gemini Pro with vision capabilities for analyzing financial charts and documents.",
        context_window=30720,
        supports_streaming=False,
        cost_per_token=0.00025,
        prompt_template="{prompt}"
    )
}

# Mistral Models
MISTRAL_MODELS = {
    "mistral_7b": ModelConfig(
        name="Mistral 7B Instruct",
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_type=ModelType.HUGGINGFACE,
        capabilities=[ModelCapability.CHAT, ModelCapability.ANALYSIS],
        max_tokens=1024,
        temperature=0.7,
        description="Mistral's efficient 7B parameter model. Fast and capable for financial conversations.",
        context_window=8192,
        supports_streaming=True,
        prompt_template="<s>[INST] {prompt} [/INST]"
    )
}

# Specialized Finance Models (Hypothetical - can be replaced with actual finance-tuned models)
FINANCE_MODELS = {
    "finbert": ModelConfig(
        name="FinBERT",
        model_id="ProsusAI/finbert",
        model_type=ModelType.HUGGINGFACE,
        capabilities=[ModelCapability.CLASSIFICATION, ModelCapability.ANALYSIS],
        max_tokens=512,
        temperature=0.1,
        description="BERT model fine-tuned on financial data for sentiment analysis and classification.",
        context_window=512,
        supports_streaming=False,
        prompt_template="Classify the financial sentiment: {prompt}"
    )
}

# Combined model registry
ALL_MODELS = {
    **GRANITE_MODELS,
    **LLAMA_MODELS,
    **FALCON_MODELS,
    **GOOGLE_MODELS,
    **MISTRAL_MODELS,
    **FINANCE_MODELS
}

# Model recommendations for specific tasks
TASK_RECOMMENDATIONS = {
    "general_chat": ["gemini", "llama2", "granite_instruct", "falcon"],
    "budget_analysis": ["gemini", "granite_instruct", "llama2_13b"],
    "spending_insights": ["gemini", "granite_instruct", "mistral_7b"],
    "financial_planning": ["gemini", "llama2_13b", "granite_instruct"],
    "code_generation": ["granite_code", "falcon_40b"],
    "data_analysis": ["gemini", "granite_code", "llama2_13b"],
    "quick_questions": ["mistral_7b", "granite_instruct", "falcon"],
    "complex_analysis": ["gemini", "llama2_13b", "falcon_40b"]
}

# Model performance tiers
PERFORMANCE_TIERS = {
    "premium": ["gemini", "llama2_13b", "falcon_40b"],
    "standard": ["granite_instruct", "llama2", "mistral_7b", "falcon"],
    "specialized": ["granite_code", "finbert"],
    "experimental": ["gemini_pro_vision"]
}

# Default model configurations
DEFAULT_CONFIGS = {
    "chat_model": "gemini",
    "analysis_model": "granite_instruct",
    "code_model": "granite_code",
    "fallback_model": "mistral_7b"
}

# API rate limits (requests per minute)
RATE_LIMITS = {
    ModelType.HUGGINGFACE: 60,
    ModelType.GOOGLE: 60,
    ModelType.OPENAI: 20,
    ModelType.LOCAL: 1000
}

# Model-specific prompt templates for different tasks
TASK_PROMPTS = {
    "budget_analysis": {
        "system": "You are a professional financial advisor specializing in budget analysis. Provide clear, actionable insights.",
        "user_template": "Analyze this budget data and provide insights:\n{data}\n\nFocus on: {focus_areas}"
    },
    
    "spending_insights": {
        "system": "You are a financial analyst expert in spending pattern recognition. Identify trends and anomalies.",
        "user_template": "Examine these spending patterns:\n{transactions}\n\nProvide insights on: {analysis_type}"
    },
    
    "savings_plan": {
        "system": "You are a certified financial planner. Create practical, achievable savings strategies.",
        "user_template": "Create a savings plan with:\nIncome: ${income}\nExpenses: ${expenses}\nGoals: {goals}"
    },
    
    "financial_advice": {
        "system": "You are a trusted financial advisor. Provide personalized, practical financial guidance.",
        "user_template": "Provide financial advice for: {situation}\nContext: {context}"
    }
}

# Error handling configurations
ERROR_CONFIGS = {
    "max_retries": 3,
    "retry_delay": 1.0,
    "timeout": 30.0,
    "fallback_enabled": True,
    "error_messages": {
        "timeout": "The model is taking longer than expected. Please try again.",
        "rate_limit": "Too many requests. Please wait a moment and try again.",
        "api_error": "There was an issue with the AI service. Trying alternative model...",
        "no_response": "The model didn't provide a response. Please rephrase your question."
    }
}

# Model validation settings
VALIDATION_CONFIGS = {
    "min_response_length": 10,
    "max_response_length": 5000,
    "required_fields": ["response", "model_used", "timestamp"],
    "prohibited_content": ["error", "failed", "unavailable"],
    "quality_threshold": 0.7
}

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model"""
    return ALL_MODELS.get(model_name)

def get_models_by_capability(capability: ModelCapability) -> List[str]:
    """Get all models that support a specific capability"""
    return [
        name for name, config in ALL_MODELS.items()
        if capability in config.capabilities
    ]

def get_recommended_models(task: str) -> List[str]:
    """Get recommended models for a specific task"""
    return TASK_RECOMMENDATIONS.get(task, ["gemini"])

def get_model_by_tier(tier: str) -> List[str]:
    """Get models in a specific performance tier"""
    return PERFORMANCE_TIERS.get(tier, [])

def validate_model_config(config: ModelConfig) -> bool:
    """Validate a model configuration"""
    required_fields = ["name", "model_id", "model_type", "capabilities", "max_tokens"]
    return all(hasattr(config, field) and getattr(config, field) is not None for field in required_fields)

def get_prompt_template(model_name: str, task: str = "chat") -> str:
    """Get the appropriate prompt template for a model and task"""
    model_config = get_model_config(model_name)
    if model_config and model_config.prompt_template:
        return model_config.prompt_template
    
    # Fallback to task-specific templates
    task_config = TASK_PROMPTS.get(task, {})
    return task_config.get("user_template", "{prompt}")

# Export commonly used configurations
__all__ = [
    'ModelConfig', 'ModelType', 'ModelCapability',
    'ALL_MODELS', 'GRANITE_MODELS', 'GOOGLE_MODELS', 'LLAMA_MODELS',
    'TASK_RECOMMENDATIONS', 'DEFAULT_CONFIGS', 'PERFORMANCE_TIERS',
    'get_model_config', 'get_models_by_capability', 'get_recommended_models',
    'get_prompt_template', 'validate_model_config'
]