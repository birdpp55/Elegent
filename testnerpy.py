import sys

sys.path.append('..')
from nerpy import NERModel

if __name__ == '__main__':
    # BertSoftmax中文实体识别模型: NERModel("bert", "shibing624/bert4ner-base-chinese")
    # BertSpan中文实体识别模型: NERModel("bertspan", "shibing624/bertspan4ner-base-chinese")
    model = NERModel("bert", "models/bertspan4ner-base-chinese")
    sentences = [
        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
        "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
    ]
    predictions, raw_outputs, entities = model.predict(sentences)
    print(entities)