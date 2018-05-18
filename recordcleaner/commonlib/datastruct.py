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
        def get_items_recursive(node, items):
            if node is None:
                return items
            items = get_items_recursive(node.left, items)
            items.append(node.data)
            items = get_items_recursive(node.right, items)
            return items

        return get_items_recursive(head, [])