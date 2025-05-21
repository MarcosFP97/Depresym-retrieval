from ollama import chat
from ollama import ChatResponse
import re

response: ChatResponse = chat(model='qwen3:latest', messages=[
  {
    'role': 'user',
    'content': 'For the of the 21 Beck Depression Inventory (BDI) questionnaire, I want you to generate 10 examples of sentences for each symptom.'\
               'The sentences need to illustrate a person experiencing the symptom and talking about themselves, they can be either positive (having the symptom) or negative (not having the symptom).' \
               'For example, for Pessimism symptom sentences could be "I feel hopeless all the time" or "I am not pessimistic at all".' \
               'Just answer with a JSON in which the key is the symptom name and the values a list of ten sentences. There cannot be overlapping between symptoms. Sentences should illustrate only one condition.'\
                'This is the list of symptoms I want you to use: "Loss of Interest in Sex".'
  },
])

result = response['message']['content']
result = re.sub(r'<think>.*?</think>', '', result)
print(result)