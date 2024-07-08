import argparse
from fengshen import UbertPipelines

total_parser = argparse.ArgumentParser("TASK NAME")
total_parser = UbertPipelines.pipelines_args(total_parser)
args = total_parser.parse_args()

args.pretrained_model_path = "IDEA-CCNL/Erlangshen-Ubert-330M-Chinese"

test_data=[
    {
        "task_type": "抽取任务", 
        "subtask_type": "实体识别", 
        "text": "这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。", 
        "choices": [ 
            {"entity_type": "小区名字"}, 
            {"entity_type": "岗位职责"}
            ],
        "id": 0}
]

model = UbertPipelines(args)
result = model.predict(test_data)
for line in result:
    print(line)