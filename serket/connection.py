class Connection:
    __connections = []
    __forward_connections = {}
    __backward_connections = {}

    def __init__(self, parent, child, forward=[], backward=[]):
        """
        Args:
            parent (serket.Module): サーバーモジュール
            child (serket.Module): クライアントモジュール
            forward (list[str]): 未実装
        """
        self.server = parent
        self.client = child
        self.forward_nodes = forward
        self.backward_nodes = backward

    def forward(self):
        params = self.server.get_params(self.forward_nodes)
        self.client.forward_params.update_params(params)
        print(f"Connection: {self.server.name} ----> {self.client.name} (Forward ={self.forward_nodes})")

    def backward(self):
        if len(self.backward_nodes) > 0:
            params = self.client.get_params(self.backward_nodes)
            self.server.backward_params.update_params(params)
            print(f"Connection: {self.server.name} <---- {self.client.name} (Backward={self.backward_nodes})")

    @staticmethod
    def get_forward_connections(module):
        """
        Args:
            module (Module):

        Returns:
            list[Connection]

        """
        connections = Connection.__forward_connections.get(module)
        if connections is None:
            return []

        return connections

    @staticmethod
    def get_backward_connections(module):
        """
        Args:
            module (Module):

        Returns:
            list[Connection]

        """
        connections = Connection.__backward_connections.get(module)
        if connections is None:
            return []

        return connections

    @staticmethod
    def register_connection(c):
        """

        Args:
            c (Connection):

        Returns:

        """
        if Connection.is_exist(c):
            raise ValueError('"{}" -> "{}"は既に接続されています'.format(c.server, c.client))

        Connection.__connections.append(c)
        if c.server in Connection.__forward_connections:
            Connection.__forward_connections[c.server].append(c)
        else:
            Connection.__forward_connections[c.server] = [c]

        if c.client in Connection.__backward_connections:
            Connection.__backward_connections[c.client].append(c)
        else:
            Connection.__backward_connections[c.client] = [c]

    @staticmethod
    def is_exist(c):
        """

        Args:
            c (Connection):

        Returns:

        """
        if c.server in Connection.__forward_connections:
            check_dict = Connection.__forward_connections[c.server]
            if check_dict in c.client:
                return True

        return False

    def __str__(self):
        return f"Connection:\n  {self.server.name} <-> {self.client.name}\n  Parameters:\n    {self.forward_nodes} <-> {self.backward_nodes}"

    def __repr__(self):
        return self.__str__()
