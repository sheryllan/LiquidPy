from collections import Mapping


def namedtuple_with_defaults(nmtype, default_values=()):
    nmtype.__new__.__defaults__ = (None,) * len(nmtype._fields)
    if isinstance(default_values, Mapping):
        prototype = nmtype(**default_values)
    else:
        prototype = nmtype(*default_values)
    nmtype.__new__.__defaults__ = tuple(prototype)
    return nmtype


class TreeMap(object):
    class Node(object):
        def __init__(self, data, left, right):
            self.data = data
            self.left = left
            self.right = right
            self.level = 1

    def skew(self, node):
        if node.left is None:
            return node
        if node.level == node.left.level:
            left = node.left
            node.left = left.right
            left.right = node
            return left
        return node

    def split(self, node):
        if node.right is None or node.right.right is None:
            return node
        if node.level == node.right.right.level:
            right = node.right
            node.right = right.left
            right.left = node
            right.level += 1
            return right
        return node

    def add(self, item, node=None):
        if node is None:
            return TreeMap.Node(item, None, None)
        key, value = item
        data = node.data
        nkey, nval = data

        if key < nkey:
            node.left = self.add(item, node.left)
        elif key == nkey:  # preserving the input order
            node.right = TreeMap.Node(item, None, node.right)
        else:
            node.right = self.add(item, node.right)

        node = self.skew(node)
        node = self.split(node)
        return node

    def get_items(self, head):
        if head is not None:
            yield from self.get_items(head.left)
            yield head.data
            yield from self.get_items(head.right)


class DynamicAttrs(object):
    def __init__(self, obj=None, **kwargs):
        self.update(obj, **kwargs)

    def update(self, obj=None, insert=True, **kwargs):
        if isinstance(obj, dict):
            for key in obj:
                setattr(self, key, obj[key])

        elif obj is not None:
            for attr in dir(obj):
                val = getattr(obj, attr)
                if not callable(val) and not attr.startswith('__'):
                    setattr(self, attr, val)

        attrs = vars(self)
        for key in kwargs:
            if insert or key in attrs:
                setattr(self, key, kwargs[key])

    def __getattr__(self, item):
        return getattr(self, item, None)