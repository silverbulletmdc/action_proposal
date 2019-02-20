"""
扩充了yaml的功能。
    1. 支持 !include语法。 reference: https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another
    2. 支持 !join语法。 reference: https://stackoverflow.com/questions/5484016/how-can-i-do-string-concatenation-or-string-replacement-in-yaml
    3. 修复了自带yaml中加载浮点数的bug。（1e-3等无法被正确识别）
"""
import sys
import os
from ruamel import yaml


def my_compose_document(self):
    self.get_event()
    node = self.compose_node(None, None)
    self.get_event()
    # self.anchors = {}    # <<<< commented out
    return node


yaml.SafeLoader.compose_document = my_compose_document


# adapted from http://code.activestate.com/recipes/577613-yaml-include-support/
def yaml_include(loader, node):
    root_dir = os.path.split(loader.name)[0]
    file_path = os.path.join(root_dir, node.value)
    with open(file_path) as inputfile:
        return my_safe_load(inputfile, master=loader)


# define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


# register the tag handler
yaml.add_constructor('!join', join, Loader=yaml.SafeLoader)
yaml.add_constructor("!include", yaml_include, Loader=yaml.SafeLoader)


def my_safe_load(stream, Loader=yaml.SafeLoader, master=None):
    loader = Loader(stream)
    if master is not None:
        loader.anchors = master.anchors
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()