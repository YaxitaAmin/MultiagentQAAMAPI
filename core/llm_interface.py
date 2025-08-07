# """LLM Interface for Multi-Agent QA System"""

# import time
# import json
# import os
# from typing import Dict, Any, List, Optional, Union
# from dataclasses import dataclass
# from enum import Enum
# from abc import ABC, abstractmethod
# from loguru import logger

# try:
#     import openai
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False

# try:
#     import anthropic
#     ANTHROPIC_AVAILABLE = True
# except ImportError:
#     ANTHROPIC_AVAILABLE = False

# try:
#     import google.generativeai as genai
#     GEMINI_AVAILABLE = True
# except ImportError:
#     GEMINI_AVAILABLE = False


# class LLMProvider(Enum):
#     OPENAI = "openai"
#     ANTHROPIC = "anthropic"
#     GEMINI = "gemini"
#     MOCK = "mock"


# @dataclass
# class LLMResponse:
#     """Standardized LLM response format"""
#     content: str
#     provider: str
#     model: str
#     usage: Dict[str, Any]
#     metadata: Dict[str, Any]
#     timestamp: float
#     success: bool


# class LLMInterface(ABC):
#     """Abstract base class for LLM interfaces"""
    
#     @abstractmethod
#     def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
#         """Generate response from LLM"""
#         pass
    
#     @abstractmethod
#     def get_provider_info(self) -> Dict[str, Any]:
#         """Get provider information"""
#         pass


# class OpenAIInterface(LLMInterface):
#     """OpenAI LLM interface implementation"""
    
#     def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
#         if not OPENAI_AVAILABLE:
#             raise ImportError("OpenAI library not available")
        
#         self.model = model
#         self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
#         self.usage_tracking = {"total_tokens": 0, "total_cost": 0.0}
        
#         logger.info(f"OpenAI interface initialized with model: {model}")
    
#     def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
#         """Generate response using OpenAI API"""
#         try:
#             start_time = time.time()
            
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=kwargs.get("temperature", 0.1),
#                 max_tokens=kwargs.get("max_tokens", 4000)
#             )
            
#             content = response.choices[0].message.content
#             usage = response.usage.__dict__ if response.usage else {}
            
#             # Track usage
#             total_tokens = usage.get("total_tokens", 0)
#             self.usage_tracking["total_tokens"] += total_tokens
            
#             # Estimate cost (rough estimates)
#             cost_per_1k_tokens = {"gpt-4o-mini": 0.00015, "gpt-4o": 0.03}.get(self.model, 0.001)
#             estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
#             self.usage_tracking["total_cost"] += estimated_cost
            
#             return LLMResponse(
#                 content=content,
#                 provider="openai",
#                 model=self.model,
#                 usage=usage,
#                 metadata={
#                     "response_time": time.time() - start_time,
#                     "estimated_cost": estimated_cost,
#                     "finish_reason": response.choices[0].finish_reason
#                 },
#                 timestamp=time.time(),
#                 success=True
#             )
            
#         except Exception as e:
#             logger.error(f"OpenAI API error: {e}")
#             return LLMResponse(
#                 content=f"Error: {str(e)}",
#                 provider="openai",
#                 model=self.model,
#                 usage={},
#                 metadata={"error": str(e)},
#                 timestamp=time.time(),
#                 success=False
#             )
    
#     def get_provider_info(self) -> Dict[str, Any]:
#         """Get OpenAI provider information"""
#         return {
#             "provider": "openai",
#             "model": self.model,
#             "available": OPENAI_AVAILABLE,
#             "usage_tracking": self.usage_tracking.copy()
#         }


# class AnthropicInterface(LLMInterface):
#     """Anthropic Claude interface implementation"""
    
#     def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
#         if not ANTHROPIC_AVAILABLE:
#             raise ImportError("Anthropic library not available")
        
#         self.model = model
#         self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
#         self.usage_tracking = {"total_tokens": 0, "total_cost": 0.0}
        
#         logger.info(f"Anthropic interface initialized with model: {model}")
    
#     def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
#         """Generate response using Anthropic API"""
#         try:
#             start_time = time.time()
            
#             response = self.client.messages.create(
#                 model=self.model,
#                 max_tokens=kwargs.get("max_tokens", 4000),
#                 temperature=kwargs.get("temperature", 0.1),
#                 messages=[{"role": "user", "content": prompt}]
#             )
            
#             content = response.content[0].text if response.content else ""
#             usage = response.usage.__dict__ if hasattr(response, 'usage') else {}
            
#             # Track usage
#             input_tokens = usage.get("input_tokens", 0)
#             output_tokens = usage.get("output_tokens", 0)
#             total_tokens = input_tokens + output_tokens
#             self.usage_tracking["total_tokens"] += total_tokens
            
#             # Estimate cost
#             cost_per_1k_input = {"claude-3-5-sonnet-20241022": 0.003}.get(self.model, 0.003)
#             cost_per_1k_output = {"claude-3-5-sonnet-20241022": 0.015}.get(self.model, 0.015)
#             estimated_cost = (input_tokens / 1000 * cost_per_1k_input) + (output_tokens / 1000 * cost_per_1k_output)
#             self.usage_tracking["total_cost"] += estimated_cost
            
#             return LLMResponse(
#                 content=content,
#                 provider="anthropic",
#                 model=self.model,
#                 usage=usage,
#                 metadata={
#                     "response_time": time.time() - start_time,
#                     "estimated_cost": estimated_cost,
#                     "stop_reason": response.stop_reason if hasattr(response, 'stop_reason') else None
#                 },
#                 timestamp=time.time(),
#                 success=True
#             )
            
#         except Exception as e:
#             logger.error(f"Anthropic API error: {e}")
#             return LLMResponse(
#                 content=f"Error: {str(e)}",
#                 provider="anthropic",
#                 model=self.model,
#                 usage={},
#                 metadata={"error": str(e)},
#                 timestamp=time.time(),
#                 success=False
#             )
    
#     def get_provider_info(self) -> Dict[str, Any]:
#         """Get Anthropic provider information"""
#         return {
#             "provider": "anthropic",
#             "model": self.model,
#             "available": ANTHROPIC_AVAILABLE,
#             "usage_tracking": self.usage_tracking.copy()
#         }


# class GeminiInterface(LLMInterface):
#     """Google Gemini interface implementation"""
    
#     def __init__(self, model: str = "gemini-pro", api_key: Optional[str] = None):
#         if not GEMINI_AVAILABLE:
#             raise ImportError("Google Generative AI library not available")
        
#         self.model = model
#         genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
#         self.client = genai.GenerativeModel(model)
#         self.usage_tracking = {"total_tokens": 0, "total_cost": 0.0}
        
#         logger.info(f"Gemini interface initialized with model: {model}")
    
#     def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
#         """Generate response using Gemini API"""
#         try:
#             start_time = time.time()
            
#             generation_config = genai.types.GenerationConfig(
#                 temperature=kwargs.get("temperature", 0.1),
#                 max_output_tokens=kwargs.get("max_tokens", 4000)
#             )
            
#             response = self.client.generate_content(
#                 prompt,
#                 generation_config=generation_config
#             )
            
#             content = response.text if response.text else ""
            
#             # Gemini doesn't provide detailed usage info in all cases
#             usage = {"candidates": len(response.candidates) if response.candidates else 1}
            
#             # Estimate tokens (rough estimation)
#             estimated_tokens = len(prompt.split()) + len(content.split())
#             self.usage_tracking["total_tokens"] += estimated_tokens
            
#             # Estimate cost (rough estimate)
#             cost_per_1k_tokens = {"gemini-pro": 0.0005}.get(self.model, 0.0005)
#             estimated_cost = (estimated_tokens / 1000) * cost_per_1k_tokens
#             self.usage_tracking["total_cost"] += estimated_cost
            
#             return LLMResponse(
#                 content=content,
#                 provider="gemini",
#                 model=self.model,
#                 usage=usage,
#                 metadata={
#                     "response_time": time.time() - start_time,
#                     "estimated_cost": estimated_cost,
#                     "estimated_tokens": estimated_tokens,
#                     "safety_ratings": [rating.__dict__ for rating in response.safety_ratings] if response.safety_ratings else []
#                 },
#                 timestamp=time.time(),
#                 success=True
#             )
            
#         except Exception as e:
#             logger.error(f"Gemini API error: {e}")
#             return LLMResponse(
#                 content=f"Error: {str(e)}",
#                 provider="gemini",
#                 model=self.model,
#                 usage={},
#                 metadata={"error": str(e)},
#                 timestamp=time.time(),
#                 success=False
#             )
    
#     def get_provider_info(self) -> Dict[str, Any]:
#         """Get Gemini provider information"""
#         return {
#             "provider": "gemini",
#             "model": self.model,
#             "available": GEMINI_AVAILABLE,
#             "usage_tracking": self.usage_tracking.copy()
#         }


# class MockLLMInterface(LLMInterface):
#     """Mock LLM interface for testing and development"""
    
#     def __init__(self, model: str = "mock-llm"):
#         self.model = model
#         self.response_templates = {
#             "planning": "Based on the goal, I recommend the following steps: 1. Navigate to target screen, 2. Interact with elements, 3. Verify results",
#             "verification": "Verification completed. The action appears to have succeeded based on the expected criteria.",
#             "supervision": "Overall analysis shows good performance with room for improvement in execution efficiency.",
#             "default": "I understand the request and will provide appropriate guidance for this QA task."
#         }
#         self.usage_tracking = {"total_tokens": 0, "total_cost": 0.0}
        
#         logger.info("Mock LLM interface initialized")
    
#     def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
#         """Generate mock response"""
#         try:
#             start_time = time.time()
            
#             # Determine response type based on prompt content
#             prompt_lower = prompt.lower()
#             if "plan" in prompt_lower or "subgoal" in prompt_lower:
#                 response_type = "planning"
#             elif "verify" in prompt_lower or "criterion" in prompt_lower:
#                 response_type = "verification"
#             elif "supervise" in prompt_lower or "recommend" in prompt_lower:
#                 response_type = "supervision"
#             else:
#                 response_type = "default"
            
#             content = self.response_templates.get(response_type, self.response_templates["default"])
            
#             # Add some variability to mock responses
#             if "JSON" in prompt:
#                 content = '{"result": "mock_json_response", "status": "success"}'
            
#             # Mock usage tracking
#             estimated_tokens = len(prompt.split()) + len(content.split())
#             self.usage_tracking["total_tokens"] += estimated_tokens
            
#             # Simulate small delay
#             time.sleep(0.1)
            
#             return LLMResponse(
#                 content=content,
#                 provider="mock",
#                 model=self.model,
#                 usage={"estimated_tokens": estimated_tokens},
#                 metadata={
#                     "response_time": time.time() - start_time,
#                     "response_type": response_type,
#                     "mock": True
#                 },
#                 timestamp=time.time(),
#                 success=True
#             )
            
#         except Exception as e:
#             logger.error(f"Mock LLM error: {e}")
#             return LLMResponse(
#                 content="Mock response failed",
#                 provider="mock",
#                 model=self.model,
#                 usage={},
#                 metadata={"error": str(e)},
#                 timestamp=time.time(),
#                 success=False
#             )
    
#     def get_provider_info(self) -> Dict[str, Any]:
#         """Get mock provider information"""
#         return {
#             "provider": "mock",
#             "model": self.model,
#             "available": True,
#             "usage_tracking": self.usage_tracking.copy()
#         }


# class MultiProviderLLMInterface:
#     """Multi-provider LLM interface with fallback capabilities"""
    
#     def __init__(self, primary_provider: LLMProvider = LLMProvider.ANTHROPIC,
#                  fallback_providers: List[LLMProvider] = None):
#         self.primary_provider = primary_provider
#         self.fallback_providers = fallback_providers or [LLMProvider.OPENAI, LLMProvider.MOCK]
#         self.interfaces = {}
#         self.current_interface = None
        
#         self._initialize_interfaces()
#         logger.info(f"Multi-provider LLM interface initialized with primary: {primary_provider.value}")
    
#     def _initialize_interfaces(self):
#         """Initialize available LLM interfaces"""
#         try:
#             # Initialize primary provider
#             self.current_interface = self._create_interface(self.primary_provider)
#             self.interfaces[self.primary_provider] = self.current_interface
            
#             # Initialize fallback providers
#             for provider in self.fallback_providers:
#                 try:
#                     interface = self._create_interface(provider)
#                     self.interfaces[provider] = interface
#                 except Exception as e:
#                     logger.warning(f"Failed to initialize {provider.value}: {e}")
                    
#         except Exception as e:
#             logger.error(f"Failed to initialize primary provider {self.primary_provider.value}: {e}")
#             # Fall back to mock
#             self.current_interface = MockLLMInterface()
#             self.interfaces[LLMProvider.MOCK] = self.current_interface
    
#     def _create_interface(self, provider: LLMProvider) -> LLMInterface:
#         """Create interface for specific provider"""
#         if provider == LLMProvider.OPENAI:
#             return OpenAIInterface()
#         elif provider == LLMProvider.ANTHROPIC:
#             return AnthropicInterface()
#         elif provider == LLMProvider.GEMINI:
#             return GeminiInterface()
#         elif provider == LLMProvider.MOCK:
#             return MockLLMInterface()
#         else:
#             raise ValueError(f"Unknown provider: {provider}")
    
#     def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
#         """Generate response with fallback support"""
#         last_error = None
        
#         # Try current interface first
#         if self.current_interface:
#             try:
#                 response = self.current_interface.generate_response(prompt, **kwargs)
#                 if response.success:
#                     return response
#                 else:
#                     last_error = response.metadata.get("error", "Unknown error")
#             except Exception as e:
#                 last_error = str(e)
#                 logger.warning(f"Current interface failed: {e}")
        
#         # Try fallback interfaces
#         for provider, interface in self.interfaces.items():
#             if interface == self.current_interface:
#                 continue
                
#             try:
#                 response = interface.generate_response(prompt, **kwargs)
#                 if response.success:
#                     logger.info(f"Fallback to {provider.value} successful")
#                     self.current_interface = interface  # Switch to working interface
#                     return response
#                 else:
#                     last_error = response.metadata.get("error", "Unknown error")
#             except Exception as e:
#                 last_error = str(e)
#                 logger.warning(f"Fallback {provider.value} failed: {e}")
        
#         # If all fail, return error response
#         return LLMResponse(
#             content=f"All providers failed. Last error: {last_error}",
#             provider="multi-provider",
#             model="unknown",
#             usage={},
#             metadata={"error": last_error, "all_providers_failed": True},
#             timestamp=time.time(),
#             success=False
#         )
    
#     def get_provider_info(self) -> Dict[str, Any]:
#         """Get information about all providers"""
#         return {
#             "primary_provider": self.primary_provider.value,
#             "current_provider": self.current_interface.get_provider_info() if self.current_interface else None,
#             "available_providers": {
#                 provider.value: interface.get_provider_info() 
#                 for provider, interface in self.interfaces.items()
#             },
#             "fallback_providers": [p.value for p in self.fallback_providers]
#         }


# def create_llm_interface(provider: str = "anthropic", **kwargs) -> LLMInterface:
#     """Factory function to create LLM interface"""
#     try:
#         provider_enum = LLMProvider(provider.lower())
        
#         if provider_enum == LLMProvider.OPENAI:
#             return OpenAIInterface(**kwargs)
#         elif provider_enum == LLMProvider.ANTHROPIC:
#             return AnthropicInterface(**kwargs)
#         elif provider_enum == LLMProvider.GEMINI:
#             return GeminiInterface(**kwargs)
#         elif provider_enum == LLMProvider.MOCK:
#             return MockLLMInterface(**kwargs)
#         else:
#             logger.warning(f"Unknown provider {provider}, falling back to multi-provider")
#             return MultiProviderLLMInterface()
            
#     except ValueError:
#         logger.warning(f"Invalid provider {provider}, falling back to multi-provider")
#         return MultiProviderLLMInterface()
#     except Exception as e:
#         logger.error(f"Error creating {provider} interface: {e}, falling back to mock")
#         return MockLLMInterface()
"""
LLM Interface - Unified interface for multiple LLM providers
Provides consistent access to various language models with AMAPI integration
"""

import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from loguru import logger

from core.logger import AMAPILogger, LogCategory


class LLMProvider(Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"


class LLMModel(Enum):
    """Supported LLM models"""
    # Anthropic
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    
    # OpenAI
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    
    # Google
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    
    # Local
    LOCAL_MODEL = "local"


@dataclass
class LLMRequest:
    """LLM request structure"""
    prompt: str
    model: LLMModel
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None


@dataclass
class LLMResponse:
    """LLM response structure"""
    content: str
    model: LLMModel
    provider: LLMProvider
    tokens_used: int
    response_time: float
    confidence: float
    metadata: Dict[str, Any]
    request_id: str


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = AMAPILogger(f"LLM_{self.__class__.__name__}")
    
    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.base_url = self.config.get('base_url', 'https://api.anthropic.com')
        
        # Try to import Anthropic
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
            self.available = True
        except ImportError:
            self.logger.warning("Anthropic library not available")
            self.client = None
            self.available = False
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic Claude"""
        start_time = time.time()
        request_id = f"anthropic_{int(time.time() * 1000)}"
        
        try:
            if not self.client:
                raise Exception("Anthropic client not initialized")
            
            # Prepare messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            
            messages.append({"role": "user", "content": request.prompt})
            
            # Make API call
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=request.model.value,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=messages
            )
            
            response_time = time.time() - start_time
            
            # Extract content
            content = response.content[0].text if response.content else ""
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=LLMProvider.ANTHROPIC,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                response_time=response_time,
                confidence=0.9,  # Anthropic models are generally high confidence
                metadata={
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'model_version': response.model
                },
                request_id=request_id
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            # Return fallback response
            return LLMResponse(
                content=f"Error generating response: {str(e)}",
                model=request.model,
                provider=LLMProvider.ANTHROPIC,
                tokens_used=0,
                response_time=time.time() - start_time,
                confidence=0.0,
                metadata={'error': str(e)},
                request_id=request_id
            )
    
    async def is_available(self) -> bool:
        """Check if Anthropic is available"""
        return self.available and self.client is not None


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        
        # Try to import OpenAI
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key) if self.api_key else None
            self.available = True
        except ImportError:
            self.logger.warning("OpenAI library not available")
            self.client = None
            self.available = False
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI GPT"""
        start_time = time.time()
        request_id = f"openai_{int(time.time() * 1000)}"
        
        try:
            if not self.client:
                raise Exception("OpenAI client not initialized")
            
            # Prepare messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            
            messages.append({"role": "user", "content": request.prompt})
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=request.model.value,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            response_time = time.time() - start_time
            
            # Extract content
            content = response.choices[0].message.content if response.choices else ""
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=LLMProvider.OPENAI,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                response_time=response_time,
                confidence=0.85,
                metadata={
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'finish_reason': response.choices[0].finish_reason if response.choices else None
                },
                request_id=request_id
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content=f"Error generating response: {str(e)}",
                model=request.model,
                provider=LLMProvider.OPENAI,
                tokens_used=0,
                response_time=time.time() - start_time,
                confidence=0.0,
                metadata={'error': str(e)},
                request_id=request_id
            )
    
    async def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return self.available and self.client is not None


class LocalProvider(BaseLLMProvider):
    """Local LLM provider (fallback)"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.available = True  # Always available as fallback
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate fallback response"""
        start_time = time.time()
        request_id = f"local_{int(time.time() * 1000)}"
        
        # Simple rule-based responses for common patterns
        prompt_lower = request.prompt.lower()
        
        if 'plan' in prompt_lower or 'strategy' in prompt_lower:
            content = """Based on the request, here's a structured approach:
1. Analyze the current situation
2. Identify key objectives and constraints
3. Develop step-by-step execution plan
4. Consider potential risks and mitigation strategies
5. Define success metrics and verification methods"""
        
        elif 'execute' in prompt_lower or 'perform' in prompt_lower:
            content = """Execution approach:
1. Prepare necessary resources and environment
2. Execute actions in logical sequence
3. Monitor progress and adapt as needed
4. Handle errors and exceptions appropriately
5. Verify completion and document results"""
        
        elif 'verify' in prompt_lower or 'check' in prompt_lower:
            content = """Verification process:
1. Define clear success criteria
2. Systematically check each requirement
3. Document findings and evidence
4. Identify any discrepancies or issues
5. Provide clear pass/fail determination"""
        
        elif 'analyze' in prompt_lower or 'examine' in prompt_lower:
            content = """Analysis framework:
1. Gather relevant data and information
2. Apply appropriate analytical methods
3. Identify patterns, trends, and insights
4. Draw evidence-based conclusions
5. Present findings with supporting rationale"""
        
        else:
            content = f"""I understand you're asking about: {request.prompt[:100]}...

Here's a structured response:
1. Acknowledge the request and its context
2. Break down the problem into manageable components
3. Apply relevant knowledge and reasoning
4. Provide actionable recommendations
5. Suggest next steps for implementation

Note: This is a local fallback response. For more sophisticated analysis, please ensure proper LLM providers are configured."""
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=content,
            model=LLMModel.LOCAL_MODEL,
            provider=LLMProvider.LOCAL,
            tokens_used=len(content.split()),  # Rough token estimate
            response_time=response_time,
            confidence=0.6,
            metadata={'fallback': True, 'response_type': 'rule_based'},
            request_id=request_id
        )
    
    async def is_available(self) -> bool:
        """Local provider is always available"""
        return True


class LLMInterface:
    """
    Unified LLM Interface for AMAPI
    Manages multiple LLM providers with automatic fallback
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = AMAPILogger("LLMInterface")
        
        # Initialize providers
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self._initialize_providers()
        
        # Usage tracking
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'average_response_time': 0.0,
            'provider_usage': {provider.value: 0 for provider in LLMProvider}
        }
        
        # Model configuration
        self.default_model = LLMModel(self.config.get('default_model', 'claude-3-5-sonnet-20241022'))
        self.fallback_models = [
            LLMModel.LOCAL_MODEL
        ]
        
        self.logger.info("LLM Interface initialized")
    
    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        try:
            # Anthropic
            anthropic_config = self.config.get('anthropic', {})
            self.providers[LLMProvider.ANTHROPIC] = AnthropicProvider(anthropic_config)
            
            # OpenAI
            openai_config = self.config.get('openai', {})
            self.providers[LLMProvider.OPENAI] = OpenAIProvider(openai_config)
            
            # Local (always available)
            local_config = self.config.get('local', {})
            self.providers[LLMProvider.LOCAL] = LocalProvider(local_config)
            
            self.logger.info(f"Initialized {len(self.providers)} LLM providers")
            
        except Exception as e:
            self.logger.error(f"Error initializing providers: {e}")
    
    async def initialize(self):
        """Initialize the LLM interface"""
        try:
            # Check provider availability
            available_providers = []
            for provider_type, provider in self.providers.items():
                if await provider.is_available():
                    available_providers.append(provider_type.value)
            
            self.logger.info(f"Available LLM providers: {', '.join(available_providers)}")
            
        except Exception as e:
            self.logger.error(f"Error during LLM interface initialization: {e}")
    
    async def cleanup(self):
        """Cleanup LLM interface resources"""
        try:
            self.logger.info("LLM Interface cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during LLM cleanup: {e}")
    
    async def generate_response(self, prompt: str, model: LLMModel = None, 
                              temperature: float = 0.7, max_tokens: int = 1000,
                              system_prompt: str = None, context: Dict[str, Any] = None,
                              agent_id: str = None) -> str:
        """Generate response using the best available LLM"""
        start_time = time.time()
        
        try:
            # Use default model if not specified
            if model is None:
                model = self.default_model
            
            # Create request
            request = LLMRequest(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                context=context,
                agent_id=agent_id
            )
            
            # Try to get response from appropriate provider
            response = await self._get_response_with_fallback(request)
            
            # Update usage statistics
            self._update_usage_stats(response, time.time() - start_time)
            
            # Log the interaction
            self.logger.debug(
                f"LLM response generated",
                category=LogCategory.SYSTEM_EVENT,
                data={
                    'model': model.value,
                    'provider': response.provider.value,
                    'tokens_used': response.tokens_used,
                    'response_time': response.response_time,
                    'confidence': response.confidence,
                    'agent_id': agent_id
                }
            )
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {e}")
            self.usage_stats['failed_requests'] += 1
            
            # Return error message
            return f"Error generating response: {str(e)}"
    
    async def _get_response_with_fallback(self, request: LLMRequest) -> LLMResponse:
        """Get response with automatic fallback to other providers"""
        try:
            # Determine primary provider for the model
            primary_provider = self._get_provider_for_model(request.model)
            
            # Try primary provider first
            if primary_provider and primary_provider in self.providers:
                provider = self.providers[primary_provider]
                if await provider.is_available():
                    try:
                        response = await provider.generate_response(request)
                        if response.confidence > 0.0:  # Valid response
                            return response
                    except Exception as e:
                        self.logger.warning(f"Primary provider {primary_provider.value} failed: {e}")
            
            # Try fallback models
            for fallback_model in self.fallback_models:
                fallback_request = LLMRequest(
                    prompt=request.prompt,
                    model=fallback_model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    system_prompt=request.system_prompt,
                    context=request.context,
                    agent_id=request.agent_id
                )
                
                fallback_provider = self._get_provider_for_model(fallback_model)
                if fallback_provider and fallback_provider in self.providers:
                    provider = self.providers[fallback_provider]
                    if await provider.is_available():
                        try:
                            response = await provider.generate_response(fallback_request)
                            self.logger.info(f"Using fallback provider: {fallback_provider.value}")
                            return response
                        except Exception as e:
                            self.logger.warning(f"Fallback provider {fallback_provider.value} failed: {e}")
            
            # If all else fails, return error response
            raise Exception("All LLM providers failed")
            
        except Exception as e:
            self.logger.error(f"Error in fallback chain: {e}")
            raise
    
    def _get_provider_for_model(self, model: LLMModel) -> Optional[LLMProvider]:
        """Get the appropriate provider for a model"""
        model_provider_map = {
            LLMModel.CLAUDE_3_5_SONNET: LLMProvider.ANTHROPIC,
            LLMModel.CLAUDE_3_HAIKU: LLMProvider.ANTHROPIC,
            LLMModel.GPT_4_TURBO: LLMProvider.OPENAI,
            LLMModel.GPT_4: LLMProvider.OPENAI,
            LLMModel.GPT_3_5_TURBO: LLMProvider.OPENAI,
            LLMModel.GEMINI_PRO: LLMProvider.GOOGLE,
            LLMModel.GEMINI_PRO_VISION: LLMProvider.GOOGLE,
            LLMModel.LOCAL_MODEL: LLMProvider.LOCAL
        }
        
        return model_provider_map.get(model)
    
    def _update_usage_stats(self, response: LLMResponse, total_time: float):
        """Update usage statistics"""
        try:
            self.usage_stats['total_requests'] += 1
            
            if response.confidence > 0.0:
                self.usage_stats['successful_requests'] += 1
            else:
                self.usage_stats['failed_requests'] += 1
            
            self.usage_stats['total_tokens'] += response.tokens_used
            self.usage_stats['provider_usage'][response.provider.value] += 1
            
            # Update average response time
            current_avg = self.usage_stats['average_response_time']
            total_requests = self.usage_stats['total_requests']
            
            self.usage_stats['average_response_time'] = (
                (current_avg * (total_requests - 1) + total_time) / total_requests
            )
            
        except Exception as e:
            self.logger.error(f"Error updating usage stats: {e}")
    
    async def get_available_models(self) -> List[LLMModel]:
        """Get list of available models"""
        available_models = []
        
        try:
            for provider_type, provider in self.providers.items():
                if await provider.is_available():
                    # Map providers to their models
                    if provider_type == LLMProvider.ANTHROPIC:
                        available_models.extend([
                            LLMModel.CLAUDE_3_5_SONNET,
                            LLMModel.CLAUDE_3_HAIKU
                        ])
                    elif provider_type == LLMProvider.OPENAI:
                        available_models.extend([
                            LLMModel.GPT_4_TURBO,
                            LLMModel.GPT_4,
                            LLMModel.GPT_3_5_TURBO
                        ])
                    elif provider_type == LLMProvider.LOCAL:
                        available_models.append(LLMModel.LOCAL_MODEL)
            
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return [LLMModel.LOCAL_MODEL]  # Always return local as fallback
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        try:
            success_rate = (
                self.usage_stats['successful_requests'] / 
                max(1, self.usage_stats['total_requests'])
            )
            
            return {
                'usage_stats': self.usage_stats.copy(),
                'success_rate': success_rate,
                'average_tokens_per_request': (
                    self.usage_stats['total_tokens'] / 
                    max(1, self.usage_stats['total_requests'])
                ),
                'provider_distribution': {
                    provider: count / max(1, self.usage_stats['total_requests'])
                    for provider, count in self.usage_stats['provider_usage'].items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating usage statistics: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers"""
        health_status = {
            'overall_health': True,
            'providers': {},
            'available_models': [],
            'recommendations': []
        }
        
        try:
            for provider_type, provider in self.providers.items():
                is_available = await provider.is_available()
                health_status['providers'][provider_type.value] = {
                    'available': is_available,
                    'status': 'healthy' if is_available else 'unavailable'
                }
                
                if not is_available and provider_type != LLMProvider.LOCAL:
                    health_status['overall_health'] = False
                    health_status['recommendations'].append(
                        f"Check {provider_type.value} configuration and API keys"
                    )
            
            health_status['available_models'] = [
                model.value for model in await self.get_available_models()
            ]
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {
                'overall_health': False,
                'error': str(e),
                'recommendations': ['Investigate LLM interface health check failure']
            }


__all__ = [
    "LLMInterface",
    "LLMProvider",
    "LLMModel",
    "LLMRequest", 
    "LLMResponse"
]