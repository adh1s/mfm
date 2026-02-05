class Linear:
    def __init__(self, t_max):
        self.t_max = t_max

    def get_coefficients(self, t):
        return 1 - t, t
