from datasets import load_dataset

# data = load_dataset('Sunbird/salt', 'text-all', split='test', trust_remote_code=True)
data = load_dataset('davidstap/NTREX', 'eng_Latn', split='dev', trust_remote_code=True)
print(data)
print(data[:2])


def doc_to_text(doc, lang):
    output = """"You are an advanced Translator, a specialized assistant designed to translate documents from 
    {source_lang} into {target_lang}. \nYour main goal is to ensure translations are grammatically correct and human-oriented.
     \n{source_lang}: {source_sentence} \n{target_lang}: """

    text = output.format(
        subject=doc["target"][lang]
    )
    return text
