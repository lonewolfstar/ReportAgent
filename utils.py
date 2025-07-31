import os

def get_tree_string(directory, prefix="", relative_path="", tree_data=None):
    if tree_data is None:
        tree_data = []
    # 获取目录中的所有文件和文件夹
    entries = os.listdir(directory)
    # 对文件和文件夹进行排序
    entries.sort()
    # 遍历每个文件和文件夹
    for i, entry in enumerate(entries):
        # 构造完整路径
        path = os.path.join(directory, entry)
        # 构造相对路径
        new_relative_path = os.path.join(relative_path, entry) if relative_path else entry
        # 判断是否为最后一个元素
        is_last = (i == len(entries) - 1)
        # 构造前缀
        if is_last:
            tree_data.append(prefix + "└── " + entry)
            new_prefix = prefix + "    "
        else:
            tree_data.append(prefix + "├── " + entry)
            new_prefix = prefix + "│   "
        # 如果是目录，递归处理
        if os.path.isdir(path):
            get_tree_string(path, new_prefix, new_relative_path, tree_data)

    return '\n'.join(tree_data)


def get_all_file_paths(directory):
    """
    遍历指定目录下的所有文件，返回每个文件的绝对路径列表

    参数:
        directory: 要遍历的文件夹路径

    返回:
        包含所有文件绝对路径的列表
    """
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            # 获取文件的完整路径
            filepath = os.path.join(root, filename)
            # 获取绝对路径并添加到列表
            file_paths.append(os.path.abspath(filepath))

    return file_paths