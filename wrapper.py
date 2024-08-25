import os
import requests
import json
from typing import List, Optional
from pydantic import BaseModel
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation

api_key = os.environ.get('DEEP_SEEK_API')
url = "https://api.deepseek.com/chat/completions"


payload = json.dumps({
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "Hi",
      "role": "user"
    }
  ],
  "model": "deepseek-coder",
  "frequency_penalty": 0,
  "max_tokens": 2048,
  "presence_penalty": 0,
  "response_format": {
    "type": "text"
  },
  "stop": None,
  "stream": False,
  "stream_options": None,
  "temperature": 1,
  "top_p": 1,
  "tools": None,
  "tool_choice": "none",
  "logprobs": False,
  "top_logprobs": None
})

class DeepSeekLLM(BaseLLM):

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        responses = []
        for prompt in prompts:
            response = requests.request("POST", url, headers=headers, data=payload)

            if response.status_code == 200:
                generated_text = response.json().get('choices')[0].get('message').get('content')
                print(generated_text)

                # 如果 stop 参数被设置，则使用它来截断生成的文本
                if stop:
                    for stop_word in stop:
                        generated_text = generated_text.split(stop_word)[0]
                
                responses.append(Generation(text=generated_text))
            else:
                responses.append(Generation(text=f"Error: {response.status_code}, {response.text}"))
        
        return LLMResult(generations=[responses])
