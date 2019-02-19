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


yaml.add_constructor("!include", yaml_include, Loader=yaml.SafeLoader)


def my_safe_load(stream, Loader=yaml.SafeLoader, master=None):
    loader = Loader(stream)
    if master is not None:
        loader.anchors = master.anchors
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()