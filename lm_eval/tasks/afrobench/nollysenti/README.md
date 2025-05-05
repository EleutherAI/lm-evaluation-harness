#

## Paper
Title: `NollySenti: Leveraging Transfer Learning and Machine Translation for Nigerian Movie Sentiment Classification`

Paper Link: https://aclanthology.org/2023.acl-short.85/

## Abstract
>Africa has over 2000 indigenous languages but they are under-represented in NLP research due to lack of datasets. In recent years, there have been progress in developing labelled corpora for African languages. However, they are often available in a single domain and may not generalize to other domains. In this paper, we focus on the task of sentiment classification for cross-domain adaptation. We create a new dataset, Nollywood movie reviews for five languages widely spoken in Nigeria (English, Hausa, Igbo, Nigerian Pidgin, and Yoruba). We provide an extensive empirical evaluation using classical machine learning methods and pre-trained language models. By leveraging transfer learning, we compare the performance of cross-domain adaptation from Twitter domain, and cross-lingual adaptation from English language. Our evaluation shows that transfer from English in the same target domain leads to more than 5% improvement in accuracy compared to transfer from Twitter in the same language. To further mitigate the domain difference, we leverage machine translation from English to other Nigerian languages, which leads to a further improvement of 7% over cross-lingual evaluation. While machine translation to low-resource languages are often of low quality, our analysis shows that sentiment related words are often preserved.

HomePage: https://github.com/IyanuSh/NollySenti

### Citation

```
@inproceedings{shode-etal-2023-nollysenti,
    title = "{N}olly{S}enti: Leveraging Transfer Learning and Machine Translation for {N}igerian Movie Sentiment Classification",
    author = "Shode, Iyanuoluwa  and
      Adelani, David Ifeoluwa  and
      Peng, JIng  and
      Feldman, Anna",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.85/",
    doi = "10.18653/v1/2023.acl-short.85",
    pages = "986--998",
    abstract = "Africa has over 2000 indigenous languages but they are under-represented in NLP research due to lack of datasets. In recent years, there have been progress in developing labelled corpora for African languages. However, they are often available in a single domain and may not generalize to other domains. In this paper, we focus on the task of sentiment classification for cross-domain adaptation. We create a new dataset, Nollywood movie reviews for five languages widely spoken in Nigeria (English, Hausa, Igbo, Nigerian Pidgin, and Yoruba). We provide an extensive empirical evaluation using classical machine learning methods and pre-trained language models. By leveraging transfer learning, we compare the performance of cross-domain adaptation from Twitter domain, and cross-lingual adaptation from English language. Our evaluation shows that transfer from English in the same target domain leads to more than 5{\%} improvement in accuracy compared to transfer from Twitter in the same language. To further mitigate the domain difference, we leverage machine translation from English to other Nigerian languages, which leads to a further improvement of 7{\%} over cross-lingual evaluation. While machine translation to low-resource languages are often of low quality, our analysis shows that sentiment related words are often preserved."
}
```
