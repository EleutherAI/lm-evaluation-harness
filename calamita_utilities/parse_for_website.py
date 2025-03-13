import os
import json
from pathlib import Path


'''
The data structure is quite complex, but still manageable:
Here an example of the target JSON file:
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
'''

class Task:
    registry={}
    def __init__(self,task_name):
         self.task_name=task_name
         self.pretty_name=""
         self.localURL=""
         self.hasPublicData=0
         self.isTranslated=0
         Task.registry[task_name]=self
    def to_dict(self):
        return {
            "task_name": self.task_name,
            "task_pretty_name": self.pretty_name,
            "task_localURL": self.localURL,
            "hasPublicData": self.hasPublicData,
            "isTranslated": self.isTranslated
        }
    @classmethod
    def get_all_data(cls):
        return [task.to_dict() for task in cls.registry.values()]   
    @classmethod
    def get_object(cls,task_name):
        if task_name in cls.registry.values():
            return cls.registry[task_name]
        else:
            return None
           
class Model:
    registry={}
    def __init__(self,model_name):
         self.model_name=model_name
         self.model_pretty_name=""
         self.model_URL=""
         Model.registry[model_name]=self
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "model_pretty_name": self.model_pretty_name,
            "model_URL": self.model_URL
        }
    @classmethod
    def get_all_data(cls):
        return [model.to_dict() for model in cls.registry.values()]   
    @classmethod
    def get_object(cls,model_name):
        if model_name in cls.registry.values():
            return cls.registry[model_name]
        else:
            return None
            
#I create a class, called Subtask, containing all the private methods to update the evaluations of a subtask
class Subtask:
    registry = {} #To avoid having two subtasks with the same name, so if it exists, it mean that in the cycle I need to update it instead of recreating it
    def __init__ (self, subtask_name, belongTo=""):
           if subtask_name in Subtask.registry:
                raise ValueError(f"'{subtask_name}' already present")
           self.subtask_name=subtask_name
           self.belongTo=belongTo
           self.evaluations=[] #Here we will store evaluations
           Subtask.registry[subtask_name]=self
    def add_evaluation(self, model_name, results):
           data={
                "model_name":model_name,
                "metrics":results
            }
           self.evaluations.append(data)

    def to_dict(self):
        return {
            "subtask_name": self.subtask_name,
            "belongTo": self.belongTo,
            "evaluations": self.evaluations
        }

    @classmethod
    def get_all_data(cls):
        return [subtask.to_dict() for subtask in cls.registry.values()]       
    @classmethod   
    def find_class(cls,name):
        return cls.registry[name]

#A function that looks for all the files in the subdirectories having *result*json in the filename
#This is intended to load files from lm_eval_harness framework
def find_result():
    result_files = {}
    for root, _, files in os.walk('.'):
        for file in files:
            if 'result' in file and file.endswith('.json'):
                path = str(Path(root) / file)
                result_files[file] = path
    return result_files

print("Cerco tutti file *result*json a partire dalla cartella attuale")

def parse_subtask(data,subtask,model,belongTo=None):
    try:
        subtask_object = Subtask(subtask_name=subtask, belongTo=belongTo)
    except ValueError:
        subtask_object=Subtask.find_class(subtask)
    metrics_tempdict=[]
    for metric in data['results'][subtask].keys():
    #Lm_eval_harness puts, in this dictionary, an "alias" having the name of the task that we don't need as well as an error measure that now we don't need
      if 'alias' not in metric and 'stderr' not in metric: 
        metric_name=metric.split(',')[0]
        metric_result=data['results'][subtask][metric]
        #print(f"DEBUG Model: {model}, Group: {belongTo}, Task: {subtask} Metric: {metric_name}, Result: {metric_result}")
        metrics_tempdict.append({"metric":metric_name,"result":metric_result})   
    subtask_object.add_evaluation(model,metrics_tempdict)


result_files=find_result()

#Each found file is now parsed 
for file in result_files:
   print(f"Parsing {file}...")
   with open(result_files[file],"r") as f:
      data=json.load(f)
      model=data['model_name']
      #Addin the model as a model 
      if Model.get_object(model) is None:
        model_object=Model(model)
      for group in data['group_subtasks'].keys(): #First I get all the groups
          if len(data['group_subtasks'][group])==0:
            '''
            If the group is empty, it means that the task doesn't have any subtask
            In this case the task is "lonely" and I directly look for a subtask with the same name of the group
            '''
            subtask=group
            parse_subtask(data,subtask,model)

          else:
             '''
              In this case, instead, the task has more than one subtask. I iterate on them to get the results
             '''
             #Adding the group as a "task"
             if Task.get_object(group) is None:
                group_object=Task(group)
             for subtask in data['group_subtasks'][group]:
                  parse_subtask(data,subtask,model,belongTo=group)

data={
"tasks":Task.get_all_data(),
"subtask":Subtask.get_all_data(),
"model":Model.get_all_data()
} 
with open("parsed.json","w") as f:
    f.write(json.dumps(data))
