import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

n = 10
pocet_vzoriek = 1000
pocet_neuronov = 200

X = torch.randn(n, pocet_vzoriek)
W = torch.randn(pocet_neuronov, n)

Z = W @ X
Z_bar = (W / n**0.5) @ X

#už iba vykreslíme histogramy
def vykresli(ax, data, nazov, farba):
    hodnoty = data.reshape(-1)
    E = hodnoty.mean().item()
    Var = hodnoty.var().item()

    print(f"{nazov}")
    print(f"E = {E:.5f}")
    print(f"Var = {Var:.5f}")
    print()

    ax.hist(hodnoty.tolist(), bins=100, density=True, color=farba)
    ax.set_xlim(-10, 10)

    ax.text(0.03, 0.05, f"E({nazov}) = {E:.4f}\nVar({nazov}) = {Var:.4f}", transform=ax.transAxes, verticalalignment="bottom")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

vykresli(ax[0], X, "x", "royalblue")
vykresli(ax[1], Z, "z", "dodgerblue")
vykresli(ax[2], Z_bar, "z", "seagreen",)

plt.tight_layout()
plt.show()
