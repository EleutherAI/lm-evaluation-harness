## Plot usage

Execute main.py with `--custom_info` to add additional columns useful for plotting.   
Once we have json results, we can plot them to wandb.   

Combine jsons to lmplot object.    
```
from lm_eval.plot import collect_jsons
lmp = collect_jsons('eval_jsons_dir')
```
    
1. Select any of the model, task or metric as (str or list). If any of these is None, all will selected. (Default: None)   
2. Can avoid writing complete task names. task="math" will plot all tasks containing keyword "math".   
```
lmp.lineplot(x="step", model=["1.3B_dedup", "1.3B"], task="math", metric="acc", compare=True,
project="my-project", name="my-run") # wandb args
```
3. Set, `compare=True` (default: False) if you want to compare models across different tasks and metrics.    
Otherwise, seperate model plots will be logged.    
4. Get filtered dataframe for any model, task and metric.   
```
lmp.filter_df(x='step', model=["19M", "19M_dedup"], , task="math", metric="acc", save_csv="19M.csv")
```   
5. Get raw dataframe 
```
lmp.get_df()
```   