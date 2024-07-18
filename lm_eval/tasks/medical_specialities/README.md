
# Medical Specialities

### Dataset

This dataset is designed for medical language models evaluation. It merges several of the most important medical QA datasets into a common format and classifies them into 35 distinct medical categories. This structure enables users to identify any specific categories where the model's performance may be lacking and address these areas accordingly.

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

* medical_specialities: Evaluates medical questions and joins by medical speciality

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