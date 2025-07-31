import re
import io
import os
import json
import base64
from PIL import Image
from tqdm import tqdm
from typing import List
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.heic', '.tif'
}


def get_image_files(folder_path: str) -> List[str]:
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return []

    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, file))

    return image_files


def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64


def remove_json_markers(s):
    # 使用正则表达式匹配开头的```json和结尾的```
    pattern = r'^```json\s*(.*?)\s*```$'
    match = re.search(pattern, s, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        # 如果没有匹配到标记，返回原始字符串（或根据需要处理）
        return s


def summarize_images(llm, img_path):
    img_base64 = image_to_base64(img_path)

    message = HumanMessage(
        content=[
            {'type': 'text',
             'text': f"""
             图片路径解析：从图像路径{img_path}中解析出一些信息，例如模型名称、训练集或者测试集等信息。
             """ + """
             理解图片内容，首先判断图像类型：
                1. 如果是实验结果图：详细分析实验图的结果，尤其关注图中的数值反映的含义。总结实验结果，并生成相应的实验结论。示例：该图展示了训练集上BERT模型的混淆矩阵。分析结果为：消极评论（0）识别效果良好，有53091个样本被正确识别，但有6880个消极评论样本被误分为积极评论（1）；积极评论（1）识别效果良好，有52805个样本被正确识别，但有7160个积极评论样本被误分为消极评论（0）。
                    输出格式示例：{
                        'understand': '该图是bert模型在测试集的接收者操作特征曲线（ROC）图。ROC曲线是一种用来评估二分类模型性能的重要工具，它通过绘制真阳性率（TPR）与假阳性率（FPR）之间的关系来展示模型的性能...',
                        'analysis': '从图中可以看出，ROC曲线整体呈现出一个良好的上升趋势，且靠近左上角，表明模型在不同阈值下都能保持较高的真阳性率和较低的假阳性率。具体来说：...',
                        'conclusion': '基于上述分析，可以得出以下结论：...',
                        'info': '模型：bert 数据集：测试集'
                    }
                 
                2. 如果是算法框架图：理解算法流程的整体框架，详细描述图中每一块区域，并按照sci的格式给出算法框架描述，能够直接用在文章里。
                    输出格式示例：{
                        'understand': '该图是算法框架图，展示了一个用于病理图像分析的算法框架图。图中详细描述了从数字化显微镜图像（WSI）到最终预测得分的整个流程 ...'
                        'analysis': '**图A**：首先对数字化的显微镜图像（WSI）进行分块处理，每个分块大小为`n x 224 x 224`。然后通过CNN层和Swin Transformer进行特征提取，并通过全局池化得到最终的特征向量（`n x 768`）。这部分流程可以总结为：...',
                        'conclusion': '基于上述算法框架图的描述，可以得出以下结论：该算法框架通过多步骤的特征提取、图结构构建和特征聚合，能够有效地从数字化显微镜图像中提取关键信息，并通过预测头输出分类结果...'，
                        'info': '算法框架图'
                    }
             返回格式：纯json格式，去掉首尾 ```json 和 ```
             """
             },
            {
                'type': 'image_url',
                'image_url': {'url': f"data:image/jpeg;base64,{img_base64}"}
            }
        ]
    )
    response = llm.invoke([message])

    return response.content


if __name__ == "__main__":
    # 加载大模型
    llm = ChatOpenAI(
        model='qwen-vl-plus',
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
        openai_api_key='sk-d08ca9f46d8f4a93bf0cafbb9b221ca8',
        temperature=0.3
    )

    # 加载实验图
    info = {}
    folder_path = "./upload_files/analysis"
    image_files = get_image_files(folder_path)
    for i, image_file in enumerate(tqdm(image_files, desc="Processing tasks")):
        summarize = summarize_images(llm, image_file)
        summarize = remove_json_markers(summarize)
        data = json.loads(summarize)
        data.update({'path': image_file})
        info.update({'image_' + str(i + 1): data})

    # 保存json
    with open('./images.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
