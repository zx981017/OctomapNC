class OctoNode:
    def __init__(self):
        """
        初始化一个 OctoNode 节点。
        """
        self.children = []
        self.data = None  # 节点数据（可以存储占用概率或其他信息）
        self.key = []  # 节点的关键字（用于唯一标识节点）

class OctoTree:
    def __init__(self,root,info):
        """
        初始化一个 OctoTree 树。
        :param root: 树的根节点
        :param info: 树的其他信息（如分辨率、大小等）
        """
        self.root = root
        self.info = info

    def get_node_by_depth(self):

        node_by_depth = {}
        keys_by_depth = {}
        self.get_node_by_depth_iter(self.root,0,node_by_depth,keys_by_depth)

        return keys_by_depth,node_by_depth

    def get_node_by_depth_iter(self, node, depth, node_by_depth, keys_by_depth):
        """
        递归遍历树，按深度存储节点。
        :param node: 当前节点
        :param depth: 当前深度
        :param node_by_depth: 存储节点的字典
        """
        if depth not in node_by_depth:
            node_by_depth[depth] = []
        if depth not in keys_by_depth:
            keys_by_depth[depth] = []
        node_by_depth[depth].append(node.data)
        keys_by_depth[depth].append(node.key)

        for child in node.children:
            if child is not None:
                self.get_node_by_depth_iter(child, depth + 1, node_by_depth, keys_by_depth)

    def get_expand_depth_nodes(self, depth):
        
        depth_nodes = []
        self.get_expand_depth_nodes_iter(self.root, depth, 0, depth_nodes)
        return depth_nodes
    
    def get_expand_depth_nodes_iter(self, node, target_depth, current_depth, depth_nodes):
        """
        递归遍历树，获取指定深度的节点。
        :param node: 当前节点
        :param target_depth: 目标深度
        :param current_depth: 当前深度
        :param depth_nodes: 存储目标深度节点的列表
        """
        for child in node.children:
            if child is not None:
                if current_depth == target_depth:
                    depth_nodes.append(node.data)
                self.get_expand_depth_nodes_iter(child, target_depth, current_depth + 1, depth_nodes)
        return
    
    def get_devide_depth_nodes(self, devide_depth, target_depth):
        """
        获取指定深度的节点。
        :param devide_depth: 指定的深度
        :return: 深度节点列表
        """
        depth_nodes = {}
        self.current_idx = -1
        self.get_devide_depth_nodes_iter(self.root, devide_depth, target_depth, 0, depth_nodes)
        return depth_nodes
    
    def get_devide_depth_nodes_iter(self, node, devide_depth, target_depth, current_depth, depth_nodes):
        
        if current_depth == devide_depth:
            self.current_idx += 1
            depth_nodes[self.current_idx] = []
        if current_depth == target_depth:
            depth_nodes[self.current_idx].append(node.data)
        for child in node.children:
            if child is not None:
                self.get_devide_depth_nodes_iter(child, devide_depth, target_depth, current_depth + 1, depth_nodes)
        return
    