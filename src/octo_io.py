import struct
from octomap import OctoNode, OctoTree


def read_header(file):
    header_info = {
            "id": None,
            "size": None,
            "res": None,
        }
    while True:
        line = file.readline()  # 读取一行
        line = (line.strip()).decode("utf-8")  # 解码为字符串

        # 跳过注释行
        if line.startswith("#"):
            continue

        # 解析关键字段
        if line.startswith("id"):
            header_info["id"] = line.split(" ", 1)[1]
        elif line.startswith("size"):
            header_info["size"] = int(line.split(" ", 1)[1])
        elif line.startswith("res"):
            header_info["res"] = float(line.split(" ", 1)[1])
        elif line == "data":
            # 到达数据部分，停止读取
            break

    return file,header_info

def read_node_recursive(file,node,key):

    value = struct.unpack('f', file.read(4))[0]
    child = struct.unpack('B', file.read(1))[0]
    node.data = value
    for k in key:
        node.key.append(k)
    for i in range(8):
        if child & (1 << i):
            child_node = OctoNode()
            node.children.append(child_node)
            child_key = []
            child_key.append((key[0] << 1) + (i & 0b1))
            child_key.append((key[1] << 1) + ((i >> 1) & 0b1))
            child_key.append((key[2] << 1) + ((i >> 2) & 0b1))
            read_node_recursive(file, child_node, child_key)
        else:
            node.children.append(None)

def read_tree(file_path):

    with open(file_path, "rb") as file:

        file,header = read_header(file)
        
        root = OctoNode()
        key = [0b0,0b0,0b0]
        read_node_recursive(file,root,key)
    
    tree = OctoTree(root,header)
    
    return tree

def read_binary_node_recursive(file, node):
    first_four_child = struct.unpack('B', file.read(1))[0]
    second_four_child = struct.unpack('B', file.read(1))[0]
    all_child = second_four_child + (first_four_child << 8)
    node.data = all_child
    for i in range(0,16,2):
        child = all_child >> (14-i)
        all_child = all_child & ~(0b11 << (14-i))
        if child == 0b11:
            child_node = OctoNode()
            node.children.append(child_node)
            read_binary_node_recursive(file,child_node)
        else:
            node.children.append(None)

def read_binary_tree(file_path):
    with open(file_path, "rb") as file:
        # Read the header
        header = read_header(file)
        # Read the tree data
        root = OctoNode()
        read_binary_node_recursive(file, root)
    return OctoTree(root, header)

if __name__ == "__main__":

    file_path = "fr2cm"

    tree = read_binary_tree(file_path)
    node_by_depth = tree.get_node_by_depth()
    for depth, nodes in node_by_depth.items():
        print(f"Depth {depth}: {len(nodes)} nodes")