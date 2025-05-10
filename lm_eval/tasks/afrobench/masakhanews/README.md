#

## Paper
Title: `MasakhaNEWS: News Topic Classification for African languages`

Paper Link: https://aclanthology.org/2023.ijcnlp-main.10/

## Abstract
>African languages are severely under-represented in NLP research due to lack of datasets covering several NLP tasks. While there are individual language specific datasets that are being expanded to different tasks, only a handful of NLP tasks (e.g. named entity recognition and machine translation) have standardized benchmark datasets covering several geographical and typologically-diverse African languages. In this paper, we develop MasakhaNEWS -- a new benchmark dataset for news topic classification covering 16 languages widely spoken in Africa. We provide an evaluation of baseline models by training classical machine learning models and fine-tuning several language models. Furthermore, we explore several alternatives to full fine-tuning of language models that are better suited for zero-shot and few-shot learning such as cross-lingual parameter-efficient fine-tuning (like MAD-X), pattern exploiting training (PET), prompting language models (like ChatGPT), and prompt-free sentence transformer fine-tuning (SetFit and Cohere Embedding API). Our evaluation in zero-shot setting shows the potential of prompting ChatGPT for news topic classification in low-resource African languages, achieving an average performance of 70 F1 points without leveraging additional supervision like MAD-X. In few-shot setting, we show that with as little as 10 examples per label, we achieved more than 90% (i.e. 86.0 F1 points) of the performance of full supervised training (92.6 F1 points) leveraging the PET approach.

HomePage: https://github.com/masakhane-io/masakhane-news

### Citation

```
@inproceedings{adelani-etal-2023-masakhanews,
    title = "{M}asakha{NEWS}: News Topic Classification for {A}frican languages",
    author = "Adelani, David Ifeoluwa  and
      Masiak, Marek  and
      Azime, Israel Abebe  and
      Alabi, Jesujoba  and
      Tonja, Atnafu Lambebo  and
      Mwase, Christine  and
      Ogundepo, Odunayo  and
      Dossou, Bonaventure F. P.  and
      Oladipo, Akintunde  and
      Nixdorf, Doreen  and
      Emezue, Chris Chinenye  and
      Al-azzawi, Sana  and
      Sibanda, Blessing  and
      David, Davis  and
      Ndolela, Lolwethu  and
      Mukiibi, Jonathan  and
      Ajayi, Tunde  and
      Moteu, Tatiana  and
      Odhiambo, Brian  and
      Owodunni, Abraham  and
      Obiefuna, Nnaemeka  and
      Mohamed, Muhidin  and
      Muhammad, Shamsuddeen Hassan  and
      Ababu, Teshome Mulugeta  and
      Salahudeen, Saheed Abdullahi  and
      Yigezu, Mesay Gemeda  and
      Gwadabe, Tajuddeen  and
      Abdulmumin, Idris  and
      Taye, Mahlet  and
      Awoyomi, Oluwabusayo  and
      Shode, Iyanuoluwa  and
      Adelani, Tolulope  and
      Abdulganiyu, Habiba  and
      Omotayo, Abdul-Hakeem  and
      Adeeko, Adetola  and
      Afolabi, Abeeb  and
      Aremu, Anuoluwapo  and
      Samuel, Olanrewaju  and
      Siro, Clemencia  and
      Kimotho, Wangari  and
      Ogbu, Onyekachi  and
      Mbonu, Chinedu  and
      Chukwuneke, Chiamaka  and
      Fanijo, Samuel  and
      Ojo, Jessica  and
      Awosan, Oyinkansola  and
      Kebede, Tadesse  and
      Sakayo, Toadoum Sari  and
      Nyatsine, Pamela  and
      Sidume, Freedmore  and
      Yousuf, Oreen  and
      Oduwole, Mardiyyah  and
      Tshinu, Kanda  and
      Kimanuka, Ussen  and
      Diko, Thina  and
      Nxakama, Siyanda  and
      Nigusse, Sinodos  and
      Johar, Abdulmejid  and
      Mohamed, Shafie  and
      Hassan, Fuad Mire  and
      Mehamed, Moges Ahmed  and
      Ngabire, Evrard  and
      Jules, Jules  and
      Ssenkungu, Ivan  and
      Stenetorp, Pontus",
    editor = "Park, Jong C.  and
      Arase, Yuki  and
      Hu, Baotian  and
      Lu, Wei  and
      Wijaya, Derry  and
      Purwarianti, Ayu  and
      Krisnadhi, Adila Alfa",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2023",
    address = "Nusa Dua, Bali",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-main.10/",
    doi = "10.18653/v1/2023.ijcnlp-main.10",
    pages = "144--159"
}
```
