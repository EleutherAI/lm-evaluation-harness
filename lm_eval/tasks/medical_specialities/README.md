
# Medical Specialities

### Dataset

This dataset is designed for medical language models evaluation. It merges several of the most important medical QA datasets into a common format and classifies them into 35 distinct medical categories. This structure enables users to identify any specific categories where the model's performance may be lacking and address these areas accordingly.

The following datasets were used in this project:

- CareQA: https://huggingface.co/datasets/HPAI-BSC/CareQA (CareQA_en.json)
- headqa_test: https://huggingface.co/datasets/openlifescienceai/headqa (test split)
- medmcqa_validation: https://huggingface.co/datasets/openlifescienceai/medmcqa (validation split)   
- medqa_4options_test: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf (test split)
- mmlu_anatomy_test: https://huggingface.co/datasets/openlifescienceai/mmlu_anatomy (test split)
- mmlu_clinical_knowledge_test: https://huggingface.co/datasets/openlifescienceai/mmlu_clinical_knowledge (test split)
- mmlu_college_medicine_test: https://huggingface.co/datasets/openlifescienceai/mmlu_college_medicine (test split)
- mmlu_medical_genetics_test: https://huggingface.co/datasets/openlifescienceai/mmlu_medical_genetics (test split)
- mmlu_professional_medicine_test: https://huggingface.co/datasets/openlifescienceai/mmlu_professional_medicine (test split)

Homepage: [Medical Specialities Dataset](https://huggingface.co/datasets/HPAI-BSC/medical-specialities)

### Citation

@misc{gururajan2024aloe,
      title={Aloe: A Family of Fine-tuned Open Healthcare LLMs}, 
      author={Ashwin Kumar Gururajan and Enrique Lopez-Cuena and Jordi Bayarri-Planas and Adrian Tormos and Daniel Hinjos and Pablo Bernabeu-Perez and Anna Arias-Duart and Pablo Agustin Martin-Torres and Lucia Urcelay-Ganzabal and Marta Gonzalez-Mallo and Sergio Alvarez-Napagao and Eduard Ayguadé-Parra and Ulises Cortés Dario Garcia-Gasulla},
      year={2024},
      eprint={2405.01886},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

### Groups and Tasks

#### Groups

* medical_specialities: Evaluates the classification of medical questions based on their respective medical specialties.

#### Tasks
  - Cardiology
  - Hematology
  - Oncology
  - Endocrinology
  - Respiratory
  - Allergy
  - Dermatology
  - Nephrology
  - Gastroenterology
  - Rheumatology
  - Otorhinolaryngology
  - Anesthesiology
  - Biochemistry
  - Pharmacology
  - Psychiatry
  - Microbiology
  - Physiology
  - Pathology
  - Obstetrics
  - Gynecology
  - Surgery
  - Emergency
  - Orthopedics
  - Neurology
  - Urology
  - Anatomy
  - Genetics
  - Radiology
  - Ophthalmology
  - Odontology
  - Pediatrics
  - Geriatrics
  - Nursing
  - Chemistry
  - Psychology


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation?

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?