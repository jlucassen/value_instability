from openai import OpenAI
import tiktoken
import os

class LLM:
    def __init__(self, model_name):
        self.openai = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name
    
    def completion(self, prompt):
        chat_completion = self.openai.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content
    
    def choice_completion(self, prompt, choices):
        choice_tokens = []
        for choice in choices:
            token = self.encoding.encode(choice)
            if len(token) > 1:
                raise ValueError(f"Choice \"{choice}\" is not encodable as a single token")
            choice_tokens += token
        logit_bias = {str(token): 100 for token in choice_tokens}
        chat_completion = self.openai.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=1,
            temperature=0, # greedy
            logit_bias=logit_bias
        )
        return chat_completion.choices[0].message.content