import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)
n, pocet_vzoriek, pocet_neuronov = 10, 1000, 200
X = torch.randn(n, pocet_vzoriek)
W = torch.randn(pocet_neuronov, n)
Z = W @ X
Z_bar = (W / n**0.5) @ X

def vykresli(ax, data, nazov, farba, titul):
    hodnoty = data.reshape(-1)
    E = hodnoty.mean().item()
    Var = hodnoty.var().item()
    print(f"{nazov}\nE = {E:.5f}\nVar = {Var:.5f}\n")
    ax.hist(hodnoty.tolist(), bins=100, density=True, color=farba)
    ax.set_xlim(-10, 10)
    ax.set_title(titul)
    ax.text(0.03, 0.05, 
        f"E({nazov}) = {E:.4f}\nVar({nazov}) = {Var:.4f}",
        transform=ax.transAxes
        )

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
vykresli(ax[0],X,"x","royalblue","Vstup X")
vykresli(ax[1],Z,"z","dodgerblue","Výstup bez škálovania")
vykresli(ax[2],Z_bar,"z_bar","seagreen","Výstup so škálovaním")
plt.tight_layout()
plt.show()