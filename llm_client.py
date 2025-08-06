import os
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class LLMConfig:
    """Configuration for LLM providers"""
    def __init__(self):
        # LLM Provider and Model (from environment)
        self.provider = os.getenv('LLM_PROVIDER', 'gemini').lower()
        self.model = os.getenv('LLM_MODEL', 'gemini-2.5-flash')
        
        # API Keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')

class LLMClient:
    """Client for making LLM API calls with provider configured from environment"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client with provider from environment
        
        Args:
            config: Optional LLMConfig object (uses environment by default)
        """
        self.config = config or LLMConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required API keys are present for the configured provider"""
        if self.config.provider == "openai" and not self.config.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        elif self.config.provider == "claude" and not self.config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        elif self.config.provider == "gemini" and not self.config.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        elif self.config.provider not in ["openai", "claude", "gemini"]:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text response (normal output)
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the specific provider
            
        Returns:
            Generated text response
        """
        if self.config.provider == "openai":
            return self._call_openai(prompt, **kwargs)
        elif self.config.provider == "claude":
            return self._call_claude(prompt, **kwargs)
        elif self.config.provider == "gemini":
            return self._call_gemini(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def generate_structured(self, prompt: str, response_schema: BaseModel, **kwargs) -> BaseModel:
        """
        Generate structured response using Pydantic schema
        
        Args:
            prompt: The input prompt
            response_schema: Pydantic model defining the expected response structure
            **kwargs: Additional parameters for the specific provider
            
        Returns:
            Parsed response matching the schema
        """
        if self.config.provider == "openai":
            return self._call_openai_structured(prompt, response_schema, **kwargs)
        elif self.config.provider == "claude":
            return self._call_claude_structured(prompt, response_schema, **kwargs)
        elif self.config.provider == "gemini":
            return self._call_gemini_structured(prompt, response_schema, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _call_openai(self, prompt: str, **kwargs) -> str:
        """Call OpenAI API for text generation"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.config.openai_api_key)
            
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    def _call_openai_structured(self, prompt: str, response_schema: BaseModel, **kwargs) -> BaseModel:
        """Call OpenAI API for structured generation"""
        try:
            from openai import OpenAI
            import json
            
            client = OpenAI(api_key=self.config.openai_api_key)
            
            # Add schema to prompt
            schema_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
            
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": schema_prompt}],
                response_format={"type": "json_object"},
                **kwargs
            )
            
            response_text = response.choices[0].message.content
            response_dict = json.loads(response_text)
            
            return response_schema.model_validate(response_dict)
            
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI structured API call failed: {str(e)}")
    
    def _call_claude(self, prompt: str, **kwargs) -> str:
        """Call Claude API for text generation"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            
            response = client.messages.create(
                model=self.config.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
            )
            
            return response.content[0].text
            
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        except Exception as e:
            raise Exception(f"Claude API call failed: {str(e)}")
    
    def _call_claude_structured(self, prompt: str, response_schema: BaseModel, **kwargs) -> BaseModel:
        """Call Claude API for structured generation"""
        try:
            import anthropic
            import json
            
            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            
            # Add schema to prompt
            schema_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
            
            response = client.messages.create(
                model=self.config.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=[{"role": "user", "content": schema_prompt}],
                **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
            )
            
            response_text = response.content[0].text
            response_dict = json.loads(response_text)
            
            return response_schema.model_validate(response_dict)
            
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        except Exception as e:
            raise Exception(f"Claude structured API call failed: {str(e)}")
    
    def _call_gemini(self, prompt: str, **kwargs) -> str:
        """Call Gemini API for text generation"""
        try:
            from google import genai
            
            client = genai.Client()
            
            response = client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                **kwargs
            )
            
            return response.text
            
        except ImportError:
            raise ImportError("Google GenAI library not installed. Run: pip install google-genai")
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    def _call_gemini_structured(self, prompt: str, response_schema: BaseModel, **kwargs) -> BaseModel:
        """Call Gemini API for structured generation"""
        try:
            from google import genai
            
            client = genai.Client()
            
            response = client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
                **kwargs
            )
            
            return response.parsed
            
        except ImportError:
            raise ImportError("Google GenAI library not installed. Run: pip install google-genai")
        except Exception as e:
            raise Exception(f"Gemini structured API call failed: {str(e)}")

# Global client instance
_llm_client = None

def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

def generate_text(prompt: str, **kwargs) -> str:
    """Quick function to generate text using configured provider"""
    client = get_llm_client()
    return client.generate_text(prompt, **kwargs)

def generate_structured(prompt: str, response_schema: BaseModel, **kwargs) -> BaseModel:
    """Quick function to generate structured output using configured provider"""
    client = get_llm_client()
    return client.generate_structured(prompt, response_schema, **kwargs) 