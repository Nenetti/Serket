import torch

from gmm.gmm import GMM
from pixyz.distributions import MixtureModel, Categorical, Normal


class TestGMM(GMM):

    def __init__(self, name, k, n_dim, device="cuda"):
        distributions = []
        for i in range(k):
            loc = torch.randn(n_dim)
            scale = torch.empty(n_dim).fill_(0.6)
            normal = Normal(name=f"{name}-p_{i:d}", var=["x"], loc=loc, scale=scale)
            distributions.append(normal)

        probs = torch.empty(k).fill_(1.0 / k)
        prior = Categorical(name=f"{name}-p_{{prior}}", var=["z"], probs=probs)
        model = MixtureModel(name="p", distributions=distributions, prior=prior).to(device)

        super(TestGMM, self).__init__(name=name, obs_nodes=["z"], model=model, prior=prior)
