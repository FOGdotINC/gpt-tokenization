import transformers

# Load the GPT tokenizer
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

# Sample text to tokenize
text = "hello world"

# Tokenize the text
input_ids = tokenizer.encode(text, return_tensors='pt')

# Convert the tokens to a list of integers
input_ids = input_ids[0].tolist()
print(input_ids)

# Split the text into words
words = text.split()

# Print the words with their corresponding tokens
for i, word in enumerate(words):
    print(f"{word}: {input_ids[i]}")
    