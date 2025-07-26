# MultiBLiMP: A Massively Multilingual Benchmark of Linguistic Minimal Pairs

## Task Description
MultiBLiMP is a massively multilingual benchmark of linguistic minimal pairs, covering 101 languages, 6 linguistic phenomena and containing more than 125,000 minimal pairs.

* Paper: https://arxiv.org/abs/2504.02768
* GitHub Repo: https://github.com/jumelet/multiblimp/
* Hugging Face Dataset Repo: https://huggingface.co/datasets/jumelet/multiblimp

## Implementation

* `multiblimp_{lang}` runs MultiBLiMP for a given language, where `{lang}` must be replaced by the language's ISO 639-3 code (e.g., `eng` for English, `abk` for Abkhazian, `wbp` for Warlpiri, etc.).
* `multiblimp` tag runs MultiBLiMP for all languages

Note: The original implementation is provided [here](https://github.com/jumelet/multiblimp), and the [dataset repository](https://huggingface.co/datasets/jumelet/multiblimp) also includes a link to a more flexible version of the implementation [here](https://github.com/catherinearnett/multiblimp). This implementation follows these as closely as possible, but the original implementations normalize length by number of tokens, which is not supported by the Language Model Evaluation Harness (see [[1](https://blog.eleuther.ai/multiple-choice-normalization/)], [[2](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)], [[3](https://github.com/EleutherAI/lm-evaluation-harness/issues/1396)]). For this reason, the implementation provided here includes both the `acc` (accuracy based on comparing the unnormalized log-probability of the correct and incorrect versions of each sentence) and `acc_norm` (the same as `acc` but with sentence log-probability normalized by number of bytes) metrics.

## Dataset Details

This table (from the [Hugging Face Dataset Repo](https://huggingface.co/datasets/jumelet/multiblimp)) lists the languages covered in MultiBLiMP and the number of items for each language.

| ISO Code |      Language      |   n  |
|:--------:|:------------------:|:----:|
| abk      | Abkhazian          | 40   |
| aqz      | Akuntsu            | 14   |
| sqi      | Albanian           | 243  |
| amh      | Amharic            | 112  |
| grc      | Ancient Greek      | 3695 |
| hbo      | Ancient Hebrew     | 983  |
| apu      | Apurinã            | 28   |
| hye      | Armenian           | 1415 |
| eus      | Basque             | 273  |
| bel      | Belarusian         | 2570 |
| ben      | Bengali            | 21   |
| bho      | Bhojpuri           | 34   |
| bor      | Borôro             | 241  |
| bre      | Breton             | 260  |
| bul      | Bulgarian          | 2458 |
| bua      | Buriat             | 103  |
| cat      | Catalan            | 2284 |
| chu      | Church Slavonic    | 4166 |
| xcl      | Classical Armenian | 1623 |
| ces      | Czech              | 4256 |
| dan      | Danish             | 50   |
| nld      | Dutch              | 2331 |
| egy      | Egyptian (Ancient) | 22   |
| eng      | English            | 770  |
| myv      | Erzya              | 464  |
| est      | Estonian           | 2575 |
| fao      | Faroese            | 232  |
| fin      | Finnish            | 2570 |
| fra      | French             | 2548 |
| glg      | Galician           | 753  |
| kat      | Georgian           | 204  |
| deu      | German             | 2298 |
| aln      | Gheg Albanian      | 677  |
| got      | Gothic             | 1579 |
| guj      | Gujarati           | 7    |
| heb      | Hebrew             | 2330 |
| azz      | H-P Nahuatl        | 207  |
| hin      | Hindi              | 1447 |
| hit      | Hittite            | 50   |
| hun      | Hungarian          | 845  |
| isl      | Icelandic          | 2801 |
| gle      | Irish              | 28   |
| ita      | Italian            | 2999 |
| quc      | K'iche'            | 131  |
| xnr      | Kangri             | 86   |
| krl      | Karelian           | 260  |
| kxh      | Karo (Ethiopia)    | 120  |
| kaz      | Kazakh             | 173  |
| kir      | Kirghiz            | 185  |
| koi      | Komi-Permyak       | 43   |
| kpv      | Komi-Zyrian        | 320  |
| lat      | Latin              | 3149 |
| lav      | Latvian            | 3032 |
| lij      | Ligurian           | 254  |
| lit      | Lithuanian         | 1180 |
| olo      | Livvi              | 190  |
| nds      | Low German         | 1774 |
| mkd      | Macedonian         | 39   |
| mar      | Marathi            | 460  |
| frm      | Middle French      | 294  |
| ell      | Modern Greek       | 1096 |
| mdf      | Moksha             | 82   |
| yrl      | Nhengatu           | 720  |
| pcm      | Nigerian Pidgin    | 26   |
| kmr      | Northern Kurdish   | 544  |
| sme      | Northern Sami      | 2536 |
| fro      | Old French         | 1976 |
| orv      | Old Russian        | 4615 |
| ota      | Ottoman Turkish    | 99   |
| fas      | Persian            | 2553 |
| xpg      | Phrygian           | 50   |
| pol      | Polish             | 3272 |
| por      | Portuguese         | 3048 |
| ron      | Romanian           | 2056 |
| rus      | Russian            | 3832 |
| san      | Sanskrit           | 4442 |
| gla      | Scottish Gaelic    | 66   |
| hbs      | Serbo-Croatian     | 3286 |
| sms      | Skolt Sami         | 263  |
| slk      | Slovak             | 4145 |
| slv      | Slovenian          | 4483 |
| spa      | Spanish            | 2541 |
| arb      | Standard Arabic    | 1215 |
| swe      | Swedish            | 201  |
| tam      | Tamil              | 382  |
| ttc      | Tektiteko          | 69   |
| tpn      | Tupinambá          | 9    |
| tur      | Turkish            | 1742 |
| uig      | Uighur             | 758  |
| ukr      | Ukrainian          | 2744 |
| hsb      | Upper Sorbian      | 186  |
| urd      | Urdu               | 550  |
| urb      | Urubú-Kaapor       | 13   |
| uzb      | Uzbek              | 50   |
| vep      | Veps               | 187  |
| wbp      | Warlpiri           | 12   |
| cym      | Welsh              | 1120 |
| hyw      | Western Armenian   | 1153 |
| wol      | Wolof              | 705  |
| sah      | Yakut              | 144  |
| nhi      | Tenango Nahuatl    | 38   |


## Citation
```
@misc{jumelet2025multiblimp10massivelymultilingual,
      title={MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs},
      author={Jaap Jumelet and Leonie Weissweiler and Arianna Bisazza},
      year={2025},
      eprint={2504.02768},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.02768},
}
```

## New Task Checklist

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
