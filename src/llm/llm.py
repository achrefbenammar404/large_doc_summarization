from typing import List, Dict, Any
from abc import ABC, abstractmethod
from openai import AzureOpenAI
from src.config.settings import get_settings
from src.logger.logger import get_logger
import time
import json 
logger = get_logger(__file__)
settings = get_settings()


class LLM(ABC):
    def __init__(self, model_name: str):
        """ 
        Args:
            model_name (str): The name of the model.
            model_provider (str): The provider or company behind the model
        """
        self.model_name = model_name
        self.client = None

    @abstractmethod
    def get_response(
        self, messages: List[str]
    ) -> str:
        """
        Args:
            messages (list):  A list of messages (conversation history).

        Returns:
            str: The generated response.
        """
        pass

    async def get_info(self) -> Dict[str, Any]:
        """
        Abstract method to return info of the LLM model.
        Returns:
            Dict[str, Any]: info.
        """
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
        }






class AzureDeployedGPT4oMiniLLM(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client: AzureOpenAI = AzureOpenAI(
            api_key=settings.GPT_4O_MINI_AZURE_API_KEY,
            base_url=settings.GPT_4O_MINI_AZURE_TARGET_URI,
            api_version=settings.GPT_4O_MINI_AZURE_API_VERSION
        )

    def get_response(
        self, 
        messages: List[str], 
        max_retries: int = 10, 
        wait_time: int = 10
    ) -> str:
        """
        Get a response from the Azure OpenAI model, with retries on failure.

        :param messages: List of message strings to send to the model.
        :param max_retries: Maximum number of retries in case of an error.
        :param wait_time: Number of seconds to wait before retrying.
        :return: The content of the response message.
        """
        attempt = 0
        #logger.debug(f"Messages are : {json.dumps(messages , indent=4)}")
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                )
                # Check if response and its content are not None
                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    raise ValueError("Received None or empty response from LLM.")
            except Exception as e:
                attempt += 1
                logger.error(f"Error occurred in get_response attempt {attempt}: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached. Unable to get a valid response. messages : {messages}")
                    return None 
class CollectionLLM : 
    llm_collection : Dict[str , LLM]= {
        'gpt-4o-mini' : AzureDeployedGPT4oMiniLLM(model_name="gpt-4o-mini")
    }



