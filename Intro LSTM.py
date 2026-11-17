import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import copy
import random

torch.manual_seed(0); np.random.seed(0)

# --- la série : 1000 points
t = np.arange(0, 100, 0.1)
serie = np.sin(t) + 0.1 * np.random.randn(len(t))

# --- découpage CHRONOLOGIQUE (jamais aléatoire sur une série temporelle)
n = len(serie)
n_train, n_val = int(0.70 * n), int(0.15 * n)

# --- normalisation avec les stats du train uniquement
mu, sigma = serie[:n_train].mean(), serie[:n_train].std()
serie = (serie - mu) / sigma

train = serie[:n_train]
val   = serie[n_train:n_train + n_val]
test  = serie[n_train + n_val:]

# --- fenêtres glissantes : 30 valeurs -> la suivante
L = 30
def fenetres(x, L):
    X = np.stack([x[i:i + L] for i in range(len(x) - L)])
    y = x[L:]
    return (torch.tensor(X[..., None], dtype=torch.float32),   # (N, L, 1)
            torch.tensor(y[:, None],   dtype=torch.float32))   # (N, 1)

Xtr, ytr = fenetres(train, L)
Xva, yva = fenetres(val, L)
Xte, yte = fenetres(test, L)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=128)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=128)

print(Xtr.shape, ytr.shape)   # torch.Size([670, 30, 1]) torch.Size([670, 1])

plt.plot(serie)
plt.plot(np.arange(n_train, n_train + n_val), val, 'r')
plt.show()

baseline = ((Xva[:, -1, :] - yva) ** 2).mean().item()
print("MSE baseline (val) :", baseline)

class PrevisionLSTM(nn.Module):
    def __init__(self, n_features=1, hidden=32, n_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,          # taille de la « mémoire »
            num_layers=n_layers,         # LSTM empilés
            batch_first=True,            # -> (batch, L, features)
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.tete = nn.Linear(hidden, 1)  # résumé -> prédiction

    def forward(self, x):                     # x : (batch, 30, 1)
        sorties, (h_n, c_n) = self.lstm(x)    # sorties : (batch, 30, hidden)
        dernier = sorties[:, -1, :]           # état après avoir lu les 30 valeurs
        return self.tete(dernier)             # (batch, 1)


def evaluer(model, loader, critere):
    model.eval()                       # désactive dropout etc.
    total, n = 0.0, 0
    with torch.no_grad():              # pas de gradient en évaluation
        for xb, yb in loader:
            total += critere(model(xb), yb).item() * len(xb)
            n += len(xb)
    return total / n

def entrainer(model, train_loader, val_loader,
              epochs=200, lr=1e-3, patience=15, clip=1.0, weight_decay=0.0):
    critere = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    meilleur, attente, meilleur_etat = float("inf"), 0, None
    hist = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        somme, n = 0.0, 0
        for xb, yb in train_loader:
            optim.zero_grad()                  # remet les gradients à 0
            perte = critere(model(xb), yb)     # 1) forward + 2) loss
            perte.backward()                   # 3) gradients (BPTT)
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # anti-explosion
            optim.step()                       # 4) mise à jour
            somme += perte.item() * len(xb); n += len(xb)

        l_train = somme / n
        l_val = evaluer(model, val_loader, critere)
        hist["train"].append(l_train); hist["val"].append(l_val)

        # --- early stopping : on garde le meilleur modèle sur la validation
        if l_val < meilleur:
            meilleur, attente = l_val, 0
            meilleur_etat = copy.deepcopy(model.state_dict())
        else:
            attente += 1
            if attente >= patience:
                print(f"Arrêt anticipé à l'epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train {l_train:.4f} | val {l_val:.4f}")

    model.load_state_dict(meilleur_etat)
    return hist

model = PrevisionLSTM(hidden=32, n_layers=1)
hist = entrainer(model, train_loader, val_loader)

# test final : UNE seule fois, à la toute fin
print("MSE test :", evaluer(model, test_loader, nn.MSELoss()))


model.eval()
with torch.no_grad():
    pred = model(Xte).squeeze().numpy()
plt.plot(yte.squeeze().numpy(), label="réel")
plt.plot(pred, "--", label="prédit")
plt.legend(); plt.show()



resultats = []
for essai in range(12):
    cfg = {
        "hidden":   random.choice([16, 32, 64, 128]),
        "n_layers": random.choice([1, 2]),
        "lr":       10 ** random.uniform(-4, -2),   # échelle logarithmique !
        "dropout":  random.choice([0.0, 0.1, 0.2]),
    }
    m = PrevisionLSTM(hidden=cfg["hidden"], n_layers=cfg["n_layers"], dropout=cfg["dropout"])
    entrainer(m, train_loader, val_loader, lr=cfg["lr"])
    score = evaluer(m, val_loader, nn.MSELoss())
    resultats.append((score, cfg))

for score, cfg in sorted(resultats, key=lambda r: r[0])[:3]:
    print(f"{score:.4f}", cfg)