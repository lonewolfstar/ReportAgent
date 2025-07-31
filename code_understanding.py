import re
import json
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader


def remove_json_markers(s):
    # 使用正则表达式匹配开头的```json和结尾的```
    pattern = r'^```json\s*(.*?)\s*```$'
    match = re.search(pattern, s, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        # 如果没有匹配到标记，返回原始字符串（或根据需要处理）
        return s


def summarize_code(llm, codes):
    map_template = """
    身份定义：你是一个强大的代码解读专家，能够准确地理解和解读代码的功能含义。
    代码片段：{code}
    任务需求：请对代码片段进行准确解读，尽量以函数或者类为单位进行解读
    返回格式：'函数名/类名': '功能描述，输入参数描述，输出参数描述，具体参数数值定义'，示例：
            ‘predict_text函数’：'根据指定的模型对输入文本进行情感分析，预测其情感类别为 “消极” 或 “积极”。函数接收待预测文本text、模型名称model_name以及包含所有模型及其相关数据的字典models作为参数。当model_name为 'bert' 或 'lstm' 时，函数会分别使用对应的模型对文本进行分词、填充和截断处理，然后通过模型获取输出并根据最大值索引返回情感类别；当model_name为其他名称时，函数会先使用 BERT 模型提取文本特征，再将这些特征输入到指定的模型中进行预测，最终返回预测的情感类别。',
            'if __name__ == '__main__'主函数': '使用 BERT 模型对文本进行情感分类训练。代码首先设置了运行环境，包括随机种子的固定以确保结果可复现，以及设备的选择（GPU 或 CPU）；接着解析命令行参数获取超参数配置，如最大序列长度、学习率、批次大小等，并创建保存模型的目录；然后加载预训练的中文 BERT 分词器和模型，以及训练和测试数据集并转换为数据加载器；之后构建基于 BERT 的分类器模型并部署到指定设备，同时定义优化器、学习率调度器和损失函数；最后进入训练循环，每个 epoch 中先训练模型并计算训练损失和准确率，再评估模型性能，更新学习率，并保存当前 epoch 的模型权重。'
            'Options类': '这段代码定义了一个名为 Options 的类，用于解析命令行参数并为模型训练配置一组完整的参数值。通过 argparse 模块，它设置了多个关键参数及其默认值：模型保存路径为 ../../results/models/lstm/，训练和测试数据集路径分别为 ../../datas/train.csv 和 ../../datas/test.csv，文本最大长度为 256，分类数量为 2，学习率为 0.001，批次大小为 8，训练轮数为 5。这些参数值共同构成了模型训练的配置方案，用户可通过命令行灵活调整。调用 Options().parse() 即可获取包含所有参数值的对象，方便在训练脚本中统一使用。'
            ......
    """
    map_prompt = PromptTemplate(template=map_template, input_variables=["code"])
    map_chain = map_prompt | llm

    information = ''
    for code in codes:
        response = map_chain.invoke({'code': code})
        information = information + '\n' + response.content

    reduce_template = """
    身份定义：你是一个强大的信息整合专家，请根据提供的描述信息进行整合
    代码信息：{information}
    任务需求：请对信息进行整合，有具体参数数值的需要进行强调，不能省略参数数值，返回示例：
        示例1：{{
            'summarize': '基于 BERT 的文本情感分类训练流程：配置环境（随机种子、设备）→解析参数→加载模型与数据→构建分类器→多轮训练（含评估、调参、保存）',
            'describe': '这段代码是一个完整的文本分类训练流程，主要功能是使用 BERT 模型对文本进行情感分类训练。代码首先设置了运行环境，包括随机种子的固定以确保结果可复现，以及设备的选择（GPU 或 CPU）；接着解析命令行参数获取超参数配置，如最大序列长度、学习率、批次大小等，并创建保存模型的目录；然后加载预训练的中文 BERT 分词器和模型，以及训练和测试数据集并转换为数据加载器；......'
        }}
        
        示例2：{{
            'summarize': 'Options 类借 argparse 解析命令行参数，含模型路径、数据集等默认配置',
            'describe': 'Options类借助argparse模块解析命令行参数，为模型训练构建完整的参数配置体系。通过__init__方法初始化参数解析器，设置的关键参数及其默认值如下：模型保存路径为../../results/models/lstm/，训练和测试数据集路径分别为../../datas/train.csv和../../datas/test.csv，文本最大长度为 256，分类数量为 2，学习率为 0.001，批次大小为 8，训练轮数为 5。'
        }}
    返回格式：纯json格式，去掉首尾 ```json 和 ```
    """
    reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["information"])
    reduce_chain = reduce_prompt | llm
    response = reduce_chain.invoke({"information": information})
    return response.content


if __name__ == "__main__":
    # 加载大模型
    llm = ChatOpenAI(
        model='qwen-turbo-latest',
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        openai_api_key='sk-d08ca9f46d8f4a93bf0cafbb9b221ca8',
        temperature=0.3
    )

    # 加载代码文档
    repo_path = "./upload_files/codes"
    loader = DirectoryLoader(
        path=repo_path,
        glob="**/*",
        loader_cls=UnstructuredFileLoader
    )
    documents = loader.load()

    # 代码文档分片
    info = {}
    for i, doc in enumerate(tqdm(documents, desc="Processing tasks")):
        python_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = python_splitter.split_documents([doc])
        summarize = summarize_code(llm, texts)
        summarize = remove_json_markers(summarize)
        data = json.loads(summarize)
        data.update({'path': doc.metadata.get("source", "")})
        info.update({'code_' + str(i + 1): data})

    # 保存json
    with open('./codes.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

    # python_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    # texts = python_splitter.split_documents([documents[1]])
    # summarize = summarize_code(llm, texts)