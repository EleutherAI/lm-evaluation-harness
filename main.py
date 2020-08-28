from gpt2 import GPT2LM

lm = GPT2LM()

print(lm.generate('1 + 1 = 2.\n3 + 5 = 8.\n4 + 9 = 13.\n4 + 3 = 7.\n2 + 3 =', '.'))