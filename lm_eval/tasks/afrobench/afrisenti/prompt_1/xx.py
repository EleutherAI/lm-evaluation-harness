from datasets import load_dataset


# ['amh', 'hau', 'ibo', 'arq', 'ary', 'yor', 'por', 'twi', 'tso', 'tir', 'orm', 'pcm', 'kin', 'swa']

data = load_dataset("masakhane/afrisenti", "pcm", trust_remote_code=True)
print(data)
print(data["test"][:5])
#
# ['Naija', 'Pipo', 'wey', 'dey', 'for', 'inside', 'social', 'Media', 'sef', 'don', 'put', 'hand', 'for', 'ear', 'give',
#  'federal', 'goment', 'and', 'polical', 'leader', 'dem', 'ova', 'di', 'kilin', '.']
#
# [6, 0, 14, 17, 2, 2, 6, 0, 7, 17, 16, 0, 2, 0, 16, 0, 0, 9, 0, 0, 11, 2, 8, 0, 1]
