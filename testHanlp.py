import hanlp

if __name__ == "__main__":
    ner = hanlp.load(
            save_dir=('./models/hanlp/mtl/msra_ner_electra_small_20220215_205503')
        )
    tok = hanlp.load(
            save_dir=('./models/hanlp/mtl/coarse_electra_small_20220616_012050')
        )
    # tokenizer = MRSATokenizer.from_pretrained("microsoft/mrsa-base")
    path="./NER/MSRA/test1.txt"
    with open(path, "r", encoding="utf-8") as file:
        dataset = file.readlines()
    sen_toks=tok(dataset)  
    ners=[]
    for sen_tok in sen_toks:
        ners.extend(ner([sen_tok],tasks='ner*'))
    output_file = "tokenized_output.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        for ner in ners:
            line = " ".join(map(str, ner))  # 将列表转换为以空格分隔的字符串
            file.write(line + "\n")

