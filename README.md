# pytorch-GNN-2nd-


<PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码

本代码还有配套的资源包
资源包需要通过www.aianaconda.com网站进行下载。


默认同步 /tmp/pycharm_project_954
我的设置 /tmp/pycharm_project_pytorch-GNN
结果2个目录都存在，而且生成的文件在/tmp/pycharm_project_pytorch-GNN


```shell
# https://blog.csdn.net/qq_40891747/article/details/116592227
pip config list
pip3 install --upgrade pip
pip3 install jupyter

pip install matplotlib

pip install transformers
pip install transformers -i https://pypi.python.org/simple

pip install spacy
python -m spacy download en_core_web_sm
```


```python

def print_variable_structure(variable, indent=""):
    print(f"{indent}Type: {type(variable)}")
    if isinstance(variable, dict):
        print(f"{indent}Keys: {list(variable.keys())}")
        for key, value in variable.items():
            print(f"{indent}Key: {key}")
            print_variable_structure(value, indent + "  ")
            break
    elif isinstance(variable, (list, tuple, np.ndarray, torch.Tensor)):
        if isinstance(variable, (list, tuple)):
            print(f"{indent}Shape: {len(variable)}")
            if len(variable) == 0:
                return
        if isinstance(variable, (np.ndarray, torch.Tensor)):
            print(f"{indent}Shape: {variable.shape}")
            print(f"{indent}Data Type: {variable.dtype}")
            if isinstance(variable, (torch.Tensor)) and variable.dim() == 0 or \
                isinstance(variable, (np.ndarray)) and variable.ndim == 0 :
                return
        for index, value in enumerate(variable):
            print(f"{indent}Index: {index}")
            print_variable_structure(value, indent + "  ")
            break
    elif hasattr(variable, "__dict__"):
        print(f"{indent}Attributes: {list(variable.__dict__.keys())}")
        for attr_name, attr_value in variable.__dict__.items():
            print(f"{indent}Attribute: {attr_name}")
            print_variable_structure(attr_value, indent + "  ")
            break


    '''
    import numpy as np
    a = np.array([
        [3, 2, 8, 4],
        [7, 2, 3, 1],
        [3, 9, 2, 4],
        [4, 1, 1, 6]
    ])
    a.argmax(1) 找出每列的最大值的索引，行号递增(不好记)
    array([2, 0, 1, 3], dtype=int64)
    for i, j in zip([0, 1, 2, 3], [2, 0, 1, 3])
    '''
```