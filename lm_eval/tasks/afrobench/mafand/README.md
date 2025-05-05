#

## Paper
Title: `A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation`

Paper Link: https://aclanthology.org/2022.naacl-main.223/

## Abstract
>Recent advances in the pre-training of language models leverage large-scale datasets to create multilingual models. However, low-resource languages are mostly left out in these datasets. This is primarily because many widely spoken languages are not well represented on the web and therefore excluded from the large-scale crawls used to create datasets. Furthermore, downstream users of these models are restricted to the selection of languages originally chosen for pre-training. This work investigates how to optimally leverage existing pre-trained models to create low-resource translation systems for 16 African languages. We focus on two questions: 1) How can pre-trained models be used for languages not included in the initial pre-training? and 2) How can the resulting translation models effectively transfer to new domains? To answer these questions, we create a new African news corpus covering 16 languages, of which eight languages are not part of any existing evaluation dataset. We demonstrate that the most effective strategy for transferring both to additional languages and to additional domains is to fine-tune large pre-trained models on small quantities of high-quality translation data.

HomePage: https://github.com/masakhane-io/lafand-mt

### Citation

```
@inproceedings{adelani-etal-2022-thousand,
    title = "A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for {A}frican News Translation",
    author = "Adelani, David  and
      Alabi, Jesujoba  and
      Fan, Angela  and
      Kreutzer, Julia  and
      Shen, Xiaoyu  and
      Reid, Machel  and
      Ruiter, Dana  and
      Klakow, Dietrich  and
      Nabende, Peter  and
      Chang, Ernie  and
      Gwadabe, Tajuddeen  and
      Sackey, Freshia  and
      Dossou, Bonaventure F. P.  and
      Emezue, Chris  and
      Leong, Colin  and
      Beukman, Michael  and
      Muhammad, Shamsuddeen  and
      Jarso, Guyo  and
      Yousuf, Oreen  and
      Niyongabo Rubungo, Andre  and
      Hacheme, Gilles  and
      Wairagala, Eric Peter  and
      Nasir, Muhammad Umair  and
      Ajibade, Benjamin  and
      Ajayi, Tunde  and
      Gitau, Yvonne  and
      Abbott, Jade  and
      Ahmed, Mohamed  and
      Ochieng, Millicent  and
      Aremu, Anuoluwapo  and
      Ogayo, Perez  and
      Mukiibi, Jonathan  and
      Ouoba Kabore, Fatoumata  and
      Kalipe, Godson  and
      Mbaye, Derguene  and
      Tapo, Allahsera Auguste  and
      Memdjokam Koagne, Victoire  and
      Munkoh-Buabeng, Edwin  and
      Wagner, Valencia  and
      Abdulmumin, Idris  and
      Awokoya, Ayodele  and
      Buzaaba, Happy  and
      Sibanda, Blessing  and
      Bukula, Andiswa  and
      Manthalu, Sam",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.223",
    doi = "10.18653/v1/2022.naacl-main.223",
    pages = "3053--3070",
    abstract = "Recent advances in the pre-training for language models leverage large-scale datasets to create multilingual models. However, low-resource languages are mostly left out in these datasets. This is primarily because many widely spoken languages that are not well represented on the web and therefore excluded from the large-scale crawls for datasets. Furthermore, downstream users of these models are restricted to the selection of languages originally chosen for pre-training. This work investigates how to optimally leverage existing pre-trained models to create low-resource translation systems for 16 African languages. We focus on two questions: 1) How can pre-trained models be used for languages not included in the initial pretraining? and 2) How can the resulting translation models effectively transfer to new domains? To answer these questions, we create a novel African news corpus covering 16 languages, of which eight languages are not part of any existing evaluation dataset. We demonstrate that the most effective strategy for transferring both additional languages and additional domains is to leverage small quantities of high-quality translation data to fine-tune large pre-trained models.",
}
```
