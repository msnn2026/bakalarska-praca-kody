import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import time

adresy = open("top10000.txt","r").read().splitlines() 
print(adresy[:10], "...\n")
print(f"Počet adries v datasete: {len(adresy)}") 

vsetky_znaky = sorted(list(set("".join(adresy))))
znak_na_index, index_na_znak = {}, {} #vytvárame slovníky
index = 1
for znak in vsetky_znaky: 
    znak_na_index[znak] = index  #... 'a':13, 'b':14, 'c':15,...
    index_na_znak[index] = znak  #... 13:'a', b:'14', 15:'c',...
    index += 1
znak_na_index["*"] = 0 # '*':0
index_na_znak[0] = "*" # 0':"*"

print("\nznak_na_index:", znak_na_index)
print("\nindex_na_znak:", index_na_znak)
pocet_znakov = len(index_na_znak)
print("\nPočet unikátnch znakov:", pocet_znakov)

def vytvor_dataset(adresy):
    pocet_predchodcov = 6
    X, Y = [], [] #trénovacia a cieľová množina
    for adresa in adresy:
        kontext = [0] * pocet_predchodcov #vektor núl (dim. 6)
        for znak in (adresa + "*"):
            index = znak_na_index[znak] #prevod na indexy
            X.append(list(kontext))  
            Y.append(index)
            kontext = kontext[1:] + [index] #posun o 1 index
    return torch.tensor(X), torch.tensor(Y)
#tu vytvárame tréningové dvojice   
X, Y = vytvor_dataset(adresy)
print(X[:14])
print(Y[:14])
print("\nTrénovacia množina:")
print("X.shape =", X.shape)
print("Y.shape =", Y.shape)

pocet_znakov = len(index_na_znak) #39 znakov
#hyperparametre
pocet_predchodcov, embedding = 6, 8
vrstva1, vrstva2 = 64, 64 #počet neurónov v jednej vrste
mini_batch, iteracie = 128, 50000
#parametre
C = torch.randn((pocet_znakov, embedding))
vstupy = embedding * pocet_predchodcov
W1 = torch.randn((vstupy, vrstva1)) * np.sqrt(2 / vstupy) 
b1 = torch.zeros(vrstva1)
W2 = torch.randn((vrstva1, vrstva2)) * np.sqrt(2 / vrstva1)
b2 = torch.zeros(vrstva2)
W3 = torch.randn((vrstva2, pocet_znakov)) * 0.01
b3 = torch.zeros(pocet_znakov)
parametre = [C, W1, b1, W2, b2, W3, b3]
for p in parametre:
    p.requires_grad_()

celkom = 0
for p in parametre:
    pocet = 1
    for i in p.shape:
        pocet = pocet * i
    celkom += pocet
print("Počet parametrov:", celkom)

random.seed(42)
random.shuffle(adresy)
Xtr, Ytr = vytvor_dataset(adresy[:int(0.85*len(adresy))])
Xval, Yval = vytvor_dataset(adresy[int(0.85*len(adresy)):])
print("Xtr.shape =", Xtr.shape, "Ytr.shape =", Ytr.shape)
print("Xval.shape =", Xval.shape, "Yval.shape =", Yval.shape)

def dopredna_propagacia(Xtr): #maticový  zápis
    A = C[Xtr] #rozšírenie mini-batchu o embeddingové vektory  
    A0 = A.reshape(A.shape[0], -1) #zreťazenie po riadkoch     
    Z1 = A0 @ W1 + b1 #broadcasting biasu b1 po riadkoch      
    A1 = torch.relu(Z1)                 
    Z2 = A1 @ W2 + b2 #broadcasting biasu b2 po riadkoch    
    A2 = torch.relu(Z2)
    Z3 = A2 @ W3 + b3 #broadcasting biasu b3 po riadkoch       
    return Z3 #toto je vektor logitov

zaciatok = time.time()
losses = []
for i in range(1, iteracie + 1):
    #náhodne vyberieme 128 indexov pre daný mini-batch
    index = torch.randint(0, Xtr.shape[0], (mini_batch,)) 
    Xmb, Ymb = Xtr[index], Ytr[index] #vyťahujeme vzorky z X a Y
    logity = dopredna_propagacia(Xmb) #dopredná propagácia
    stratova_funkcia = F.cross_entropy(logity, Ymb)
    for p in parametre: #spätná propagácia
        p.grad = None #vynulujeme staré gradienty
    stratova_funkcia.backward()
    eta = (0.001 - 0.03) / iteracie * i + 0.03 #update
    with torch.no_grad(): #tu už gradienty nepotrebujeme ukladať
        for p in parametre:
            p -= eta * p.grad
    losses.append(stratova_funkcia.item())

    if i % 5000 == 0 or i == 1:
        aktualny_cas = time.time() - zaciatok
        minuty = int(aktualny_cas // 60)
        sekundy = aktualny_cas % 60
        print(
            f"\nTréning: {100 * i / iteracie:.1f}% , "
            f"Loss: {stratova_funkcia.item():.4f}"
        )
        print("krok:", np.round(eta, 4))
        print(f"čas od začiatku: {sekundy:.1f} s")
koniec = time.time()
celkovy_cas = koniec - zaciatok
sekundy = celkovy_cas % 60
print(f"Celkový čas tréningu: {sekundy:.1f} s")

with torch.no_grad(): #vypíšeme výdledné hodnoty tr/val
    logity_train = dopredna_propagacia(Xtr)
    train_loss = F.cross_entropy(logity_train, Ytr)
    logity_val = dopredna_propagacia(Xval)
    val_loss = F.cross_entropy(logity_val, Yval)
print("train loss:", train_loss.item())
print("val loss:", val_loss.item())

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
x_vsetky = list(range(1, len(losses) + 1))
x_500 = [1] + list(range(500, len(losses) + 1, 500))
y_500 = [losses[0]] + losses[499::500]
ax[0].plot(x_vsetky, losses, color="#4c72b0")
ax[0].set_xlabel("Iterácia")
ax[0].set_ylabel("Loss")
ax[0].set_title("Vývoj stratovej funkcie počas tréningu")
ax[0].set_xlim(-2000, len(losses) + 2000)
ax[1].plot(x_500, y_500, color="#4c72b0")
ax[1].set_xlabel("Iterácia")
ax[1].set_ylabel("Loss")
ax[1].set_title("Stratová funkcia každých 500 iterácií")
ax[1].set_xlim(-2000, len(losses) + 2000)
plt.tight_layout()
plt.show()

torch.manual_seed(42)
for j in range(10):
    vystup= []
    kontext = [0] * pocet_predchodcov
    while True:
        logity = dopredna_propagacia(torch.tensor([kontext]))
        pravdep_vektor = F.softmax(logity, dim=1)
        #vyberieme 10 najväčších hodnôt
        hodnoty, indexy = torch.topk(pravdep_vektor, 10, dim=1) 
        novy_pravdep_vektor = hodnoty / hodnoty.sum(
            dim=1,
            keepdim=True
        ) #dostávame nové hodnoty
        vybrany_index = torch.multinomial(
            novy_pravdep_vektor,
            num_samples=1
        ).item()
        index = indexy[0, vybrany_index].item()
        kontext = kontext[1:] + [index]
        vystup.append(index)
        if index == 0:
            break
    text = ""  
    for i in vystup:
        znak = index_na_znak[i]
        if znak != "*":
            text = text + znak
    print(text)