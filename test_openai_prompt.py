

import openai
openai.api_key = "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"

prompt = "Write a short story about a person who can travel through time."

model_engine = "text-davinci-002"
max_tokens = 60
temperature = 0.5
n_completions = 1

response = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature,
    n=n_completions
)

completion_text = response.choices[0].text.strip()
print(completion_text)
