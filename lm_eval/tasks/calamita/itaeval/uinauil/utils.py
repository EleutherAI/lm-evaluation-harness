# See: https://github.com/valeriobasile/uinauil/blob/main/src/uinauil.py#L409C9-L414C10
LABELMAP = {
    "00": "Neutrale",
    "01": "Negativo",
    "10": "Positivo",
    "11": "Misto"
}

def process_sentipolc(dataset):
    return dataset.map(lambda x: {"polarity": LABELMAP[f"{x['opos']}{x['oneg']}"]})