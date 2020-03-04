class Connection:
    """

    """
    __connections = []
    __forward_connections = {}
    __backward_connections = {}

    def __init__(self, parent, child, shared_nodes):
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

        if c.parent in Connection.__forward_connections:
            Connection.__backward_connections[c.parent].append(c.child)
        else:
            Connection.__backward_connections[c.parent] = [c.child]

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
