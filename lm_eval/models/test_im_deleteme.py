from InterventionModel import InterventionModel
m = InterventionModel.from_csv(base_name='google/gemma-2-2b', csv_path='/home/cs29824/matthew/sae_jailbreak_unlearning/src/scripts/evaluation/lm-evaluation-harness/lm_eval/models/debug_steer.csv', device='cuda:1')
m.generate('hi')