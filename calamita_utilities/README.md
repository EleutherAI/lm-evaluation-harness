# Python Scripts for Processing lm_eval_harness Results
This folder contains useful Python scripts designed to manage the raw results coming from lm_eval_harness framework for Calamita's purposes.
- - -
## Scripts overview
### parse_for_website.py
#### Functionality
This script by default looks for each file having "result*" in the name starting from the current folder and walking in every subdirectory.
This is useful because results coming from lm_eval_harness are usually organized in several different subfolders.
#### Output Structure
The script generates an output file, "parsed.json" consisting as a json file organized in this way:

```json
{
  "tasks": [
    {
      "task_name": "",
      "task_pretty_name": "",
      "task_localURL": "",
      "hasPublicData": 0,
      "isTranslated": 0
    }
  ],
  "subtasks": [
    {
      "subtask_name": "",
      "belongTo": "",  // Empty if the subtask is independent
      "evaluations": [
        {
          "model_name": "",
          "metrics": {
            "name1": 0,
            "name2": 0
          }
        },
        {
          "model_name": "",
          "metrics": {
            "name1": 0,
            "name2": 0
          }
        }
      ]
    }
  ],
  "models": [
    {
      "model_name": "",
      "model_pretty_name": "",
      "model_URL": ""
    }
  ]
}
```
#### Integration with Calamita's website
Calamita's website is coded for automatically load a json file into this format to generate the total results table as well as adding the results for each task in the website. Be aware that not always this JSON is immediately usable into the website: some tasks until now have some naming problems, so additional post-processing using different simple scripts could be necessary to align every subtask with its parent task.
In this repo a sample "parsed.json" file is included. 

### genera_per_autori.py
#### Functionality
This script is intended to reorganize all the samples logs coming from lm_eval_harness (the folder with the evaluation is **hard-coded** as a constant in the code, (cartella_risultati='tests_calamita_latest') of course it is dirty and should be changed in a different folder structure.

**Input (Original Structure):**

```
model_name => all the samples together
```

This script copies all the files in a different directory structure, more suitable for communication with tasks' authors.
The structure after running the script will be:

**Output (New Structure):**

```
per_autori/
  ├── task_name/
  │   ├── subtask_name/
  │   │   ├── modelname_date.jsonl
```

The correspondence task/subtask is taken from the CSV file "tabellaConversioneGruppiTask.csv", the same used to automatically adapt the raw "parsed.json" file to the structure of the website.
