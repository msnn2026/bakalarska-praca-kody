import numpy as np
import matplotlib.pyplot as plt
#vstupné dáta a cieľový vektor
X = np.array([ 
    [ 1.5, -0.8,  0.3],
    [ 0.2,  1.3, -1.5],
    [-1.2,  0.4,  1.1],
    [-0.7, -1.4,  0.6],])
Y = np.array([0.9, -0.8, 0.4, 0.7])
eta, n_iter= 0.03, 300

def tanh(x): #hyperbolický tangens + príslušná derivácia
    tanh = np.tanh(x)
    d_tanh = 1 - tanh**2
    return tanh, d_tanh #indexy [0] a [1]

def sigmoid(x): #sigmoidná funkcia + príslušná derivácia
    sigmoid = 1 / (1 + np.exp(-x))
    d_sigmoid = sigmoid * (1 - sigmoid)
    return sigmoid, d_sigmoid #indexy [0] a [1]

def ReLU(x): #rectified linear unit + príslušná derivácia
    ReLU = np.maximum(0, x)
    d_ReLU = np.where(x > 0, 1.0, 0.0)
    return ReLU, d_ReLU #indexy [0] a [1]

def trening(metoda, aktivacna_f):
    #inicializácia váh
    np.random.seed(48) #náhodne vyberáme vzorky pre SGD
    W1 = np.array([[0.2, -0.1, 0.1],
                   [-0.1, 0.1, 0.2]])
    b1 = np.array([0.1, -0.1])
    W2 = np.array([[0.1, -0.2]])
    b2 = 0.2

    def dopredna_propagacia(x):
        z1 = W1 @ x + b1
        a1 = aktivacna_f(z1)[0] 
        #index [0] predstavuje akt.funkciu (tanh/Sigmoid/ReLU)
        y_hat = W2 @ a1 + b2
        return y_hat[0], z1, a1 
        #index [0] prevedie výsledok na skalár

    def stratova_f():
        L = 0.0
        for i in range(len(X)):
            y_hat = dopredna_propagacia(X[i])[0]
            L += (y_hat - Y[i])**2
        return L
    losses = []

    for k in range(n_iter + 1):
        if metoda == "SGD":
            vzorka = [np.random.randint(len(X))] #náhodny výber
        elif metoda == "GD":
            vzorka = range(len(X)) #0,1,2,3

        dW1, db1 = np.zeros_like(W1), np.zeros_like(b1)
        dW2, db2 = np.zeros_like(W2), np.zeros_like(b2)

        for i in vzorka:
            #dopredná propagácia
            y_hat, z1, a1 = dopredna_propagacia(X[i])
            #spätná propagácia
            dL_dy = 2 * (y_hat - Y[i])
            dW2 += dL_dy * a1.reshape(1, -1)
            db2 += dL_dy
            dL_da1 = dL_dy * W2[0] 
            #index [0] prevedie výsledok na vektor
            dL_dz1 = dL_da1 * aktivacna_f(z1)[1] 
            #index [0] predstavuje deriváciu aktivačnej funkcie
            dW1 += np.outer(dL_dz1, X[i])
            db1 += dL_dz1
        #aktualizácia parametrov (gradient delíme počtom vzoriek)
        W1 -= eta * dW1 / len(vzorka) 
        b1 -= eta * db1 / len(vzorka)
        W2 -= eta * dW2 / len(vzorka)
        b2 -= eta * db2 / len(vzorka)
        losses.append(stratova_f())

        if k % 50 == 0: 
        #hodnota strat. funkcie po každých 50 iteráciach
            predikcie = []
            for j in range(len(X)):
                predikcie.append(dopredna_propagacia(X[j])[0])
            print(
                f"iteracia {k}: loss = {stratova_f():.4f}, "
                f"odhad = = {np.round(predikcie,4)}"
                )
    return losses

loss_gd_tanh = trening("GD",tanh); print()
loss_gd_sigmoid = trening("GD",sigmoid); print()
loss_gd_ReLU = trening("GD",ReLU); print()
loss_sgd_tanh = trening("SGD",tanh); print()
loss_sgd_sigmoid = trening("SGD",sigmoid); print()
loss_sgd_ReLU = trening("SGD",ReLU); print()

#vykreslenie grafu
plt.figure(figsize=(10, 6))
plt.plot(loss_gd_tanh,'--',label='GD tanh',color="#6baed6")
plt.plot(loss_gd_sigmoid,'--',label='GD sigmoid',color="#fdae6b")
plt.plot(loss_gd_ReLU,'--',label='GD ReLU',color="#74c476")
plt.plot(loss_sgd_tanh,label='SGD tanh',color="#1f77b4")
plt.plot(loss_sgd_sigmoid,label='SGD sigmoid',color="#ff7f0e")
plt.plot(loss_sgd_ReLU,label='SGD ReLU',color="#2ca02c")
plt.plot([],[],' ',label='počet iterácii = 300')
plt.plot([],[],' ',label='krok učenia = 0.03') #alebo 0.1
plt.xlabel('Iterácia')
plt.ylabel('Stratová funkcia')
plt.title('Porovnanie SGD a GD pre vybrané aktivačné funkcie')
plt.legend()
plt.tight_layout()
plt.show()