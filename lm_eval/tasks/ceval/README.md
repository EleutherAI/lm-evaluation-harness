# C-Eval (Validation)

### Paper
C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models
https://arxiv.org/pdf/2305.08322.pdf

C-Eval is a comprehensive Chinese evaluation suite for foundation models.
It consists of 13948 multi-choice questions spanning 52 diverse disciplines
and four difficulty levels.

Homepage: https://cevalbenchmark.com/

### Citation

```bibtex
@article{huang2023ceval,
    title={C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models},
    author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian},
    journal={arXiv preprint arXiv:2305.08322},
    year={2023}
}
```


SUBJECTS = {
    "computer_network":"计算机网络",
    "operating_system":"操作系统",
    "computer_architecture":"计算机组成",
    "college_programming":"大学编程",
    "college_physics":"大学物理",
    "college_chemistry":"大学化学",
    "advanced_mathematics":"高等数学",
    "probability_and_statistics":"概率统计",
    "discrete_mathematics":"离散数学",
    "electrical_engineer":"注册电气工程师",
    "metrology_engineer":"注册计量师",
    "high_school_mathematics":"高中数学",
    "high_school_physics":"高中物理",
    "high_school_chemistry":"高中化学",
    "high_school_biology":"高中生物",
    "middle_school_mathematics":"初中数学",
    "middle_school_biology":"初中生物",
    "middle_school_physics":"初中物理",
    "middle_school_chemistry":"初中化学",
    "veterinary_medicine":"兽医学",
    "college_economics":"大学经济学",
    "business_administration":"工商管理",
    "marxism":"马克思主义基本原理",
    "mao_zedong_thought":"毛泽东思想和中国特色社会主义理论体系概论",
    "education_science":"教育学",
    "teacher_qualification":"教师资格",
    "high_school_politics":"高中政治",
    "high_school_geography":"高中地理",
    "middle_school_politics":"初中政治",
    "middle_school_geography":"初中地理",
    "modern_chinese_history":"近代史纲要",
    "ideological_and_moral_cultivation":"思想道德修养与法律基础",
    "logic":"逻辑学",
    "law":"法学",
    "chinese_language_and_literature":"中国语言文学",
    "art_studies":"艺术学",
    "professional_tour_guide":"导游资格",
    "legal_professional":"法律职业资格",
    "high_school_chinese":"高中语文",
    "high_school_history":"高中历史",
    "middle_school_history":"初中历史",
    "civil_servant":"公务员",
    "sports_science":"体育学",
    "plant_protection":"植物保护",
    "basic_medicine":"基础医学",
    "clinical_medicine":"临床医学",
    "urban_and_rural_planner":"注册城乡规划师",
    "accountant":"注册会计师",
    "fire_engineer":"注册消防工程师",
    "environmental_impact_assessment_engineer":"环境影响评价工程师",
    "tax_accountant":"税务师",
    "physician":"医师资格"
}


# CMMLU

### Paper

CMMLU: Measuring massive multitask language understanding in Chinese
https://arxiv.org/abs/2306.09212

CMMLU is a comprehensive evaluation benchmark specifically designed to evaluate the knowledge and reasoning abilities of LLMs within the context of Chinese language and culture.
CMMLU covers a wide range of subjects, comprising 67 topics that span from elementary to advanced professional levels.

Homepage: https://github.com/haonan-li/CMMLU

### Citation

```bibtex
@misc{li2023cmmlu,
      title={CMMLU: Measuring massive multitask language understanding in Chinese},
      author={Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin},
      year={2023},
      eprint={2306.09212},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

- `ceval-valid`: All 52 subjects of the C-Eval dataset, evaluated following the methodology in MMLU's original implementation. This implementation consists solely of the validation set of C-Eval, as the test set requires submission of model predictions to an external site.

#### Tasks


The following tasks evaluate subjects in the C-Eval dataset using loglikelihood-based multiple-choice scoring:
- `ceval-valid_{subject_english}`

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation?

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
