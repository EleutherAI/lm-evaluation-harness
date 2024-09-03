## README

##  Single prompt Tasks

## 1. Textual Entailment
Relevant files:
- `_te_template_yaml`
- `_evalita-sp_te_task.yaml`

## 2. Document Dating
Relevant files:
- `_dd_template_yaml`
- `_evalita-sp_dd_task.yaml`

## 3. Sentiment Analysis
- `_sa_template_yaml`
- `_sa_template_v2_yaml`
- `_evalita-sp_sa_task.yaml`
- `_evalita-sp_sa_task_v2.yaml`
- `utils.py`
    - lines 10-50
- `metrics.py`
    - lines 10-50

V1 and V2 differs in the evaluation metric: v2 considers the macro average weighted by support of the F1 score, while v1 is based on the evaluation described [here](https://s3.cbk.cloud.syseleven.net/elg-public/60ef3fa107dc4869a353869a5d51201b_u2235_paper_026.pdf)

## 4. Hate Speech Detection
- `_hs_template_yaml`
- `_evalita-sp_hs_task.yaml`

# 5. Lexical Substitution
- `_ls_template_yaml`
- `_evalita-sp_ls_task.yaml`
- `utils.py`
    - lines 58-98
- `metrics.py`
    - lines 17-53
The model is asked to provide synonims separated by commas and the output is parsed accordingly.

## 6. Word in Context
- `_wic_template_yaml`
- `_evalita-sp_wic_task.yaml`

## 7. Named Entity Recognition
- `_ner_template_yaml`
- `_evalita-sp_ner_task.yaml`
- `utils.py`
    - lines 147-349
- `metrics.py`
    - lines 162-180

The model is asked to provide a list of entity in the format ENT1$TYPE1%ENT2$TYPE2%... and the output is parsed accordingly. If there are no entities it is asked to output "&&NOENT&&"
If the format is changed the parsing and scoring functions in `utils.py` should be updated accordingly.

## 8. Relation Extraction
- `_re_template_yaml`
- `_evalita-sp_re_task.yaml`
- `utils.py`
    - lines 349-400
- `metrics.py`
    - lines 185-198
The format is the same as the one for NER.

## 9. FAQ
- `_faq_template_yaml`
- `_evalita-sp_faq_task.yaml`
- `utils.py`
    - lines 643-657
The dataset has two versions. V1 is the one to use

## 10. Admissions Test
- `_at_template_yaml`
- `_evalita-sp_at_task.yaml`


## 11. Headline Translation
- `_ht_template_yaml`
- `_evalita-sp_ht_task.yaml`
- `utils.py`
    - lines 660-670
- `metrics.py`
    - lines 199-263

## 12. Summarization
- `_sum_template_fp_yaml`
- `_evalita-sp_sum_fp_task.yaml`
- `_evalita-sp_sum_ip_task.yaml`
- `utils.py`
    - lines 619-642

## Multi prompt Tasks
The multi prompt tasks are evaluated using the same code as the single prompt tasks. The only difference is that there are more configuation yaml files to be used.
For each nlp task there are several yaml files:
- `_evalita-mp_{nlp-taskcode}.yaml` which defines the task group and the tasks
- `_evalita-mp_{taskcode}_{prompt_number}.yaml` which defines the single task and tag(that has to match the 'task' field in the yaml defining the group. Confusing, I know.)

Be sure that the aggregate_metric_list field in the group yaml files list the metric used to evaluate the tasks in the group. Otherwise you'll see no aggregated results.

### For NER
The tasks `evalita-mp_ner_group` aggregates all the scores from the three sub datasets and prompts.
The tasks:
- `evalita-mp_ner_fic_group`
- `evalita-mp_ner_adg_group`
- `evalita-mp_ner_wn_group`
aggregate the scores for the respective sub datasets.


## Command line parameters
- `--task`: the task to evaluate (e.g. evalita-sp or evalita-mp)
- `--model`: the model to evaluate
- `--apply_chat_template`: if the model is instruction tuned this is nedeed 
- `--system_instruction=str`: System prompt of the model
- `--num_fewshot`: number of fewshot examples to use. NB. the yaml file has to specify the split to use to get the fewshot example