import json

# `response` is the ChatCompletion you showed
msg = response.choices[0].message

# 1. This is the JSON text the model returned:
qa_json_str = msg.content

# (You can ignore msg.reasoning_content if you don't want the hidden reasoning)
# print(msg.reasoning_content)  # <- usually don't log this in production

# 2. Parse the JSON
qa_obj = json.loads(qa_json_str)

# 3. Extract the 2 QA samples
samples = qa_obj["samples"]

for i, s in enumerate(samples, start=1):
    print(f"Q{i}: {s['question']}")
    print(f"Reasoning:\n{s['reasoning_trace']}")
    print(f"Answer: {s['answer']}")
    print("-----")