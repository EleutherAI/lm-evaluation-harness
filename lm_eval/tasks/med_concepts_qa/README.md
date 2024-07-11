# MedConceptsQA

### Paper

Title: `MedConceptsQA: Open Source Medical Concepts QA Benchmark`

Abstract: https://arxiv.org/abs/2405.07348

MedConceptsQA is a dedicated open source benchmark for medical concepts question answering. The benchmark comprises of questions of various medical concepts across different vocabularies: diagnoses, procedures, and drugs.

The questions are categorized into three levels of difficulty: easy, medium, and hard.

Our benchmark serves as a valuable resource for evaluating the
abilities of Large Language Models to interpret medical codes and distinguish
between medical concepts.

### Citation

```
@article{shoham2024medconceptsqa,
  title={MedConceptsQA--Open Source Medical Concepts QA Benchmark},
  author={Shoham, Ofir Ben and Rappoport, Nadav},
  journal={arXiv preprint arXiv:2405.07348},
  year={2024}
}
```

### Groups and Tasks

#### Groups

* `med_concepts_qa`: Contains all the QA tasks (diagnosis, procedures ,and drugs).

#### Tasks


* `med_concepts_qa_icd9cm` - ICD9-CM (diagnosis codes, ICD9 format) question-answering. This involves providing information, clarifications, and answering questions related to ICD-9-CM (International Classification of Diseases, 9th Revision, Clinical Modification) diagnosis codes.


* `med_concepts_qa_icd10cm` - ICD10-CM (diagnosis codes, ICD10 format) question-answering. This involves providing information, clarifications, and answering questions related to ICD-10-CM (International Classification of Diseases, 10th Revision, Clinical Modification) diagnosis codes.


* `med_concepts_qa_icd9proc` - ICD9-Proc (procedure codes, ICD9 format) question-answering. This involves providing information, clarifications, and answering questions related to ICD-9-PCS (International Classification of Diseases, 9th Revision, Procedure Coding System) procedure codes.


* `med_concepts_qa_icd10proc` - ICD10-Proc (procedure codes, ICD10 format) question-answering. This involves providing information, clarifications, and answering questions related to ICD-10-PCS (International Classification of Diseases, 10th Revision, Procedure Coding System) procedure codes.


* `med_concepts_qa_atc` - ATC (Anatomical Therapeutic Chemical Classification System) question-answering. This involves providing information, clarifications, and answering questions related to the ATC classification system, which is used for the classification of drugs and other medical products according to the organ or system on which they act and their therapeutic, pharmacological, and chemical properties.
