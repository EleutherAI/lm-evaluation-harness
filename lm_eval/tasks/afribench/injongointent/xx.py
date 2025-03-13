from datasets import load_dataset
data = load_dataset('masakhane/InjongoIntent', 'eng', split="train")#, trust_remote_code=True)

print(data)
# print(data[:2])

if "span" in data.features:
    print(True)
else:
    print(False)
# for row in data['text'][:5]:
#     print(row)

LANGUAGES_MAPPER = {
    "amh": "Amharic",
    "ewe": "Ewe",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lin": "Lingala",
    "lug": "Luganda",
    "orm": "Oromo",
    "sna": "Shona",
    "sot": "Sotho",
    "swa": "Swahili",
    "twi": "Twi",
    "wol": "Wolof",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
    "eng": "English",
}

languages = ['amh', 'eng', 'ewe', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna', 'sot', 'swa', 'twi', 'wol',
             'xho', 'yor', 'zul']
