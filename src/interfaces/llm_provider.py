import abc
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("CBRN-Sentinel.LLM")

class LLMInterface(abc.ABC):
    @abc.abstractmethod
    async def chat(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Standard chat interface.
        Returns: {'content': str, 'raw': Any}
        """
        pass

class OpenAIProvider(LLMInterface):
    """
    Universal OpenAI-Compatible Provider.
    Works for: OpenAI, DeepSeek, Ollama, vLLM, Groq, etc.
    """
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4o"):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.model = model
        except ImportError:
            raise ImportError("pip install openai required")

    async def chat(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # "Reasoning" models (o1, gpt-5-nano?) often enforce temperature=1 or don't support it.
            # We omit it for these models to use the mandatory default.
            run_args = {
                "model": self.model,
                "messages": messages,
            }
            if "nano" not in self.model and "o1" not in self.model:
                 run_args["temperature"] = 0.0

            response = await self.client.chat.completions.create(**run_args)
            return {
                "content": response.choices[0].message.content,
                "raw": response.model_dump()
            }
        except Exception as e:
            logger.error(f"OpenAI/Compatible API Error: {e}")
            return {"content": f"Error: {str(e)}", "raw": {}}

class GoogleProvider(LLMInterface):
    """
    Google GenAI Provider (V2 SDK).
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        try:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=api_key)
            self.types = types
            self.model = "gemini-1.5-flash"
        except ImportError:
            raise ImportError("pip install google-genai required")

    async def chat(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        try:
            config = self.types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=0.0
            )
            # Run blocking call in thread
            import asyncio
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=prompt,
                config=config
            )
            return {
                "content": response.text,
                "raw": str(response)
            }
        except Exception as e:
            logger.error(f"Google API Error: {e}")
            return {"content": f"Error: {str(e)}", "raw": {}}

class LLMFactory:
    @staticmethod
    @staticmethod
    def create(provider: str, api_key: str, base_url: str = None, model: str = None) -> LLMInterface:
        provider = provider.lower()
        
        if provider == "google":
            return GoogleProvider(api_key, model=model or "gemini-1.5-flash")
        
        elif provider == "deepseek":
            return OpenAIProvider(
                api_key, 
                base_url="https://api.deepseek.com", 
                model=model or "deepseek-chat"
            )
            
        elif provider == "openai":
            return OpenAIProvider(api_key, model=model or "gpt-5-nano")
            
        elif provider == "generic":
            if not base_url:
                raise ValueError("base_url is required for generic provider")
            return OpenAIProvider(api_key, base_url=base_url, model=model or "default")
            
        else:
            raise ValueError(f"Unknown provider: {provider}")
