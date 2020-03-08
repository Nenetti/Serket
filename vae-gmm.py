from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from serket.dataset import Dataset
from test_gmm import TestGMM
from test_vae import TestVAE

encoder_dim = 128
decoder_dim = 128

DEVICE = "cuda"

root = "./data"

BATCH_SIZE = 30000


def load_dataset(transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambd=lambda x: x.view(-1))])

    train_dataset = datasets.MNIST(root=root, train=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)

    train_dataset = Dataset(x=train_dataset.data.view(-1, 784).float(), label=train_dataset.targets)
    test_dataset = Dataset(x=test_dataset.data.view(-1, 784).float(), label=test_dataset.targets)

    return train_dataset, test_dataset


def main():
    x_dim = 784
    z_dim = 2

    train_dataset, test_dataset = load_dataset()

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ############################################################################################################################################################
    # モジュール宣言
    ############################################################################################################################################################
    # VAE
    vae = TestVAE(name="TestVAE", x_dim=x_dim, z_dim=z_dim, data_loader=train_loader, device=DEVICE)
    # GMM
    gmm = TestGMM(name="TestGMM", k=10, n_dim=z_dim, device=DEVICE)

    ############################################################################################################################################################
    # モジュール接続
    ############################################################################################################################################################
    vae.connect(module=gmm, forward=["z"], backward=["loc"])

    ############################################################################################################################################################
    # 学習
    ############################################################################################################################################################
    print("Start Training")
    for i in range(100):
        vae.update(input_vars=["x"])
        gmm.update()

        """
        データxをVAEに通すことで平均μと分散Σと潜在変数z1が得られる
        このz1をGMMに通すことで各分布の平均μ'と分散Σ'と潜在変数z1の各分布に対する負担率(所属確率)が得られる
        
        ということは論文で述べられているμ_{z2}は各データにおける各分布の平均と分散を負担率で重み付けした値？
        
        
        つまり n番目のデータを x_{n}, k番目の分布(クラスター)を k, 各分布への負担率を γ_{k} とすると
        μ_{z2|x=x_{n}} = Σ_(k=1~K) ( N(μ_{k}, σ_{k}) * γ_{k} )
        
        μ_{z2}はデータごとに異なってくる(負担率による影響)のためVAE側のパラメータ更新はバッチサイズ1で対応せざる負えなくなる？
        
        """


if __name__ == "__main__":
    main()
