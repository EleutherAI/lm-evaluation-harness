# hu-collocation-lambada

## Dataset Summary
This dataset is a Hungarian benchmark designed to evaluate large language models' understanding of contextual collocations and definitions. It is inspired by the LAMBADA task and constructed using the full content of the *Magyar szókapcsolatok, kollokációk adatbázisa* (Temesi, ed.).

Each data point contains:
- a target collocation from the original dataset,
- its dictionary-style definition,
- and a short narrative ending just before the collocation's final word.

The benchmark tests whether a language model can predict the missing collocation word based on a provided definition and a contextually appropriate story.


## Example Format
```json
{
  "word": "ábrázoló művészet",
  "description": "a valóságot anyagi eszközökkel megjelenítő művészet, pl. a festészet és a szobrászat",
  "class": "general",
  "text": "Édesapám mindig is szeretett festeni. Mikor gyermek voltam, észrevettem, ahogy az ecsetvonásai új világokat teremtettek a vásznon: erdők, emberek, fények, árnyékok elevenedtek meg. Akkor értettem meg, mit jelent az ábrázoló művészet"
}
```

## During evaluation (e.g. with lm-evaluation-harness), the model is prompted as:

```plaintext
Megcélzott szókapcsolat leírása: {{description}}
Folytasd a szöveget! Válaszodban csak a helyes szót add vissza!
{{text.split(' ')[:-1]|join(' ')}}
```

The model must return the final word (or word sequence) of the collocation.

## Task and Motivation
This benchmark adapts the LAMBADA principle to Hungarian using real linguistic resources. It evaluates whether language models:

- understand definitions in Hungarian,
- resolve meaning in narrative context,
- and predict precise wordings of known collocations.

It helps benchmark models on semantic understanding, cultural fluency, and context-sensitive text prediction in Hungarian.

## Languages
- Hungarian (hu)

## Dataset Structure

| Field       | Type   | Description                                   |
|-------------|--------|-----------------------------------------------|
| word        | string | Target collocation (e.g. "ábrázoló művészet") |
| description | string | Collocation definition (from the original source) |
| class       | string | Collocation type/class ("general" or "domain") |
| text        | string | Generated story ending with the target collocation |

## Data Sources and Attribution
The target collocations and their definitions are sourced from:

- Temesi Viola (szerk.): *Magyar szókapcsolatok, kollokációk adatbázisa*.  
  Tinta Kiadó, Budapest.  
  Digitális Tankönyvtár:  
  https://dtk.tankonyvtar.hu/xmlui/bitstream/handle/123456789/8874/Magyar_szokapcsolatok_kollokaciok_adatbazisa.pdf

The narrative texts and the `class` labels were generated automatically using OpenAI’s GPT-4o based on the definitions and collocations. Prompts were designed to encourage short, coherent stories that end with the given collocation and to classify collocations into appropriate categories.

These generated contents do not appear in the original source and were created solely for research and benchmarking purposes.

Use of this dataset is restricted to non-commercial, research-focused contexts, with attribution to the original source required.


## Authors and Credits
This benchmark dataset was created by the AI/NLP team at **Graphium Ltd (Graphium Kft)**, with the goal of enriching the Hungarian LLM ecosystem and providing tools for rigorous benchmarking.

We combined structured linguistic data with generative modeling techniques to simulate meaningful narrative contexts for collocations.

- Created by: Graphium Ltd (Graphium Kft) – AI/NLP research team  
- Website: https://www.graphium.hu  
- Contact: hello@graphium.hu  

We welcome collaboration, feedback, and contributions from the NLP community.

## Licensing
- **License:** Research use only.  
- **Attribution required:** Tinta Kiadó, Temesi Viola (szerk.), Oktatási Hivatal (Digitális Tankönyvtár).  
- **Commercial use:** Not permitted without additional rights clearance.

## Citation
```bibtex
@misc{hu-coll-lambada,
  title = {Hungarian Collocation LAMBADA Benchmark},
  author = {Graphium Ltd (Graphium Kft)},
  year = {2025},
  note = {Based on Temesi (ed.): Magyar szókapcsolatok, kollokációk adatbázisa. Tinta Kiadó},
  howpublished = {\url{https://huggingface.co/datasets/graphium-company/hu-collocation-lambada}}
}