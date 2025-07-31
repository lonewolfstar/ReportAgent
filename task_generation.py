import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import Docx2txtLoader


def generation(llm, user, outline=None):
    if outline:
        template = """
        你是一个经验丰富的项目报告、学术论文、实验报告撰写者，能够根据提供的用户输入的项目背景、数据描述、写作要求、大纲模板等信息：
        项目背景：{project}
        数据描述：{dataset}
        写作要求：{requirements}
        大纲模板：{template}
        将撰写任务进行步骤分解，即为大纲的每一章节生成相应的写作步骤或者描述（尤其满足写作要求），以json格式返回结果，例如：
        {{'1.'：{{title: '背景描述', 'describe': '写作步骤或者描述', 'child': ’无‘}}, 
         '2.'：{{title: '模型构建', 'describe': '写作步骤或者描述', 'child': 
         {{'2.1.': {{title: 'BERT情感分类模型', 'describe': '写作步骤或者描述', 'child': None}}, '2.2.': {{title: 'BERT+LSTM情感分类模型', 'describe': '写作步骤或者描述', 'child': None}}}}}},
         ......}}
        返回格式：纯json格式，去掉首尾 ```json 和 ```
        """
        prompt = PromptTemplate(template=template, input_variables=['project', 'dataset', 'requirements', 'template'])
        chain = prompt | llm
        response = chain.invoke({'project': user['project'],
                                 'dataset': user['dataset'],
                                 'requirements': user['requirements'],
                                 'template': outline})
    else:
        template = """
        你是一个经验丰富的项目报告、学术论文、实验报告撰写者，能够根据提供的用户输入的项目背景、数据描述、写作要求等信息：
        项目背景：{project}
        数据描述：{dataset}
        写作要求：{requirements}
        根据项目背景、数据描述、写作要求等信息（尤其满足写作要求）生成写作大纲
        大纲应具有逻辑性和连贯性，让读者能够自然地从一个观点过渡到下一个观点。
        并将撰写任务进行步骤分解，即为大纲的每一章节生成相应的写作步骤或者描述，以json格式返回结果，例如：
        {{'1.'：{{title: '背景描述', 'describe': '写作步骤或者描述', 'child': None}}, 
         '2.'：{{title: '模型构建', 'describe': '写作步骤或者描述', 'child': 
         {{'2.1.': {{title: 'BERT情感分类模型', 'describe': '写作步骤或者描述', 'child': None}}, '2.2.': {{title: 'BERT+LSTM情感分类模型', 'describe': '写作步骤或者描述', 'child': None}}}}}},
         ......}}
        返回格式：纯json格式，去掉首尾 ```json 和 ```
        """
        prompt = PromptTemplate(template=template, input_variables=['project', 'dataset', 'requirements'])
        chain = prompt | llm
        response = chain.invoke(
            {'project': user['project'], 'dataset': user['dataset'], 'requirements': user['requirements']})
    return response.content


if __name__ == "__main__":
    # 加载大模型
    llm = ChatOpenAI(
        model='qwen-plus',
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        openai_api_key='sk-d08ca9f46d8f4a93bf0cafbb9b221ca8',
        temperature=0.3
    )

    # 用户输入描述
    user_input = {
        'project': '构建了Bert模型和Bert_LSTM模型进行商品评论情感分类，对用户的商品评论进行分类，分为消极（负类）和积极（正类）',
        'dataset': '本研究采用的多源数据集整合京东用户行为数据与微博社交媒体的公开文本，构建了一个包含超过10万+条文本样本的综合数据集。数据来源分为以下两个部分：一方面，从微博平台抓取了公开讨论内容，形成了名为weibo_senti_100k.csv的数据集。这部分数据包含了实时话题讨论和公众观点，能够反映社交媒体上的即时情感动态。另一方面通过京东商品评论，形成了名为jd_conm.csv的数据集。最后通过分层采样处理了主流新闻媒体和网络论坛中的时事评论与热点讨论，构成了两个数据集：train.csv（weibo_senti_100k）和test.csv（jd_conm清洗后数据）。',
        'requirements': '1. 背景部分字数不少于600字； 2. 模型描述部分，包括描述模型的原理尽量包含相应的公式(latex格式表示)，字数不少于100字'
    }

    # 加载代码文档
    repo_path = "./upload_files/outline/outline.docx"
    loader = Docx2txtLoader(repo_path)
    documents = loader.load()
    outline_template = documents[0].page_content

    # 生成任务
    task = generation(llm, user_input, outline_template)
    print(task)

    # 格式转换
    data = json.loads(task)
    print(data)

    # 保存json
    with open('./task.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
