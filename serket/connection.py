class Connection:
    """

    """
    __connections = []
    __forward_connections = {}
    __backward_connections = {}

    def __init__(self, parent, child, shared_nodes):
        """
        Args:
            parent (serket.Module): サーバーモジュール
            child (serket.Module): クライアントモジュール
            shared_nodes (list[str]): 共有ノード
        """
        self.parent = parent
        self.child = child
        self.shared_nodes = shared_nodes

    @staticmethod
    def register_connection(c):
        """

        Args:
            c (Connection):

        Returns:

        """
        if Connection.is_exist(c):
            raise ValueError('"{}" -> "{}"は既に接続されています'.format(c.parent, c.child))

        Connection.__connections.append(c)
        if c.parent in Connection.__forward_connections:
            Connection.__forward_connections[c.parent].append(c.child)
        else:
            Connection.__forward_connections[c.parent] = [c.child]

        if c.child in Connection.__backward_connections:
            Connection.__backward_connections[c.child].append(c.parent)
        else:
            Connection.__backward_connections[c.child] = [c.parent]

    @staticmethod
    def is_exist(c):
        """

        Args:
            c (Connection):

        Returns:

        """
        if c.parent in Connection.__forward_connections:
            check_dict = Connection.__forward_connections[c.parent]
            if check_dict in c.child:
                return True

        return False

    def __str__(self):
        return f"{self.parent.name} -> {self.child.name}"

    def __repr__(self):
        return self.__str__()
