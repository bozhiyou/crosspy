from typing import Any


class AttrRetriever:
    def __init__(self, attr: str, attr_getter_dict=None):
        """
        attr_getter_dict
            type -> getter func
        """
        self.attr_name = attr
        self.attr_getter_dict = attr_getter_dict

    def __call__(self, obj, *, attr_dict=None):
        if hasattr(obj, self.attr_name):
            return getattr(obj, self.attr_name)
        if self.attr_getter_dict and type(obj) in self.attr_getter_dict:
            return self.attr_getter_dict[type(obj)](obj)
        if attr_dict and id(obj) in attr_dict:
            return attr_dict[id(obj)]
        raise Exception("Unknown attribute '" + self.attr_name + "' for %s" % obj)

def default_shape(obj):
    if hasattr(obj, "__len__"):
        return (len(obj),)
    return None
get_shape = AttrRetriever('shape', default_shape)