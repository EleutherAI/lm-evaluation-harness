# SuperGPQA

### Paper

Title: `SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines`

Abstract: `Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable performance gap between current model capabilities and artificial general intelligence.`

Homepage: https://supergpqa.github.io/

Dataset: https://huggingface.co/datasets/m-a-p/SuperGPQA (26,529 multiple-choice questions, 13 disciplines, 72 fields, 285 subfields; most questions have 10 options, some fewer)

Reference implementation: https://github.com/SuperGPQA/SuperGPQA

This implementation follows the zero-shot setting of the reference implementation: the official zero-shot instruction (`config/prompt/zero-shot.yaml`), the official option formatting (`A) option` lines), greedy decoding, and answer extraction from the final `Answer: $LETTER` line (`group_select: -1` mirrors the reference extractor's preference for the last stated answer). The reference implementation additionally applies multi-stage fallback extraction (e.g., matching option text when no letter is found), which a single-pass regex filter does not replicate; scores for models that do not follow the requested answer format may differ slightly from the official evaluator.

### Citation

```bibtex
@misc{pteam2025supergpqascalingllmevaluation,
      title={SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines},
      author={M-A-P Team and Xinrun Du and Yifan Yao and Kaijing Ma and Bingli Wang and Tianyu Zheng and Kang Zhu and Minghao Liu and Yiming Liang and Xiaolong Jin and Zhenlin Wei and Chujie Zheng and Kaixin Deng and Shian Jia and Sichao Jiang and Yiyan Liao and Rui Li and Qinrui Li and Sirun Li and Yizhi Li and Yunwen Li and Dehua Ma and Yuansheng Ni and Haoran Que and Qiyao Wang and Zhoufutu Wen and Siwei Wu and Tianshun Xing and Ming Xu and Zhenzhu Yang and Zekun Moore Wang and Junting Zhou and Yuelin Bai and Xingyuan Bu and Chenglin Cai and Liang Chen and Yifan Chen and Chengtuo Cheng and Tianhao Cheng and Keyi Ding and Siming Huang and Yun Huang and Yaoru Li and Yizhe Li and Zhaoqun Li and Tianhao Liang and Chengdong Lin and Hongquan Lin and Yinghao Ma and Zhongyuan Peng and Zifan Peng and Qige Qi and Shi Qiu and Xingwei Qu and Yizhou Tan and Zili Wang and Chenqing Wang and Hao Wang and Yiya Wang and Yubo Wang and Jiajun Xu and Kexin Yang and Ruibin Yuan and Yuanhao Yue and Tianyang Zhan and Chun Zhang and Jingyang Zhang and Xiyue Zhang and Xingjian Zhang and Yue Zhang and Yongchi Zhao and Xiangyu Zheng and Chenghua Zhong and Yang Gao and Zhoujun Li and Dayiheng Liu and Qian Liu and Tianyu Liu and Shiwen Ni and Junran Peng and Yujia Qin and Wenbo Su and Guoyin Wang and Shi Wang and Jian Yang and Min Yang and Meng Cao and Xiang Yue and Zhaoxiang Zhang and Wangchunshu Zhou and Jiaheng Liu and Qunshu Lin and Wenhao Huang and Ge Zhang},
      year={2025},
      eprint={2502.14739},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14739},
}
```

### Groups and Tasks

#### Groups

* `supergpqa`: All 13 disciplines of the SuperGPQA dataset, evaluated in the official zero-shot setting (micro-averaged over all 26,529 questions).

#### Tasks

The following tasks evaluate individual disciplines of the SuperGPQA dataset:

- `supergpqa_agronomy`
- `supergpqa_economics`
- `supergpqa_education`
- `supergpqa_engineering`
- `supergpqa_history`
- `supergpqa_law`
- `supergpqa_literature_and_arts`
- `supergpqa_management`
- `supergpqa_medicine`
- `supergpqa_military_science`
- `supergpqa_philosophy`
- `supergpqa_science`
- `supergpqa_sociology`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
