"""
LSTM : trajectoire 2D (x_px, y_px) -> trajectoire 3D (x_m, y_m, z_m)
Version finale : un seul réseau (bidirectionnel, 2 couches, hidden 64, lr 3e-3), choisi d'après les tests.
Sorties : lstm_trajectoires.pt (poids du réseau uniquement), normalisation.npz, trajectoires_3d.png,
          trajectoires_predites.csv (toutes les trajectoires prédites par le réseau)

Usage : python trajectoires_lstm_final.py                -> entraîne, teste, sauvegarde, extrait (test), trace
        python trajectoires_lstm_final.py extraire       -> SANS réentraîner : recharge le .pt, extrait le jeu de test
        python trajectoires_lstm_final.py extraire tout  -> idem pour les 10 221 trajectoires (train, val, test)
        (on peut aussi écrire : extraire val   |   extraire train val   |   ...)
"""
import copy, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0); np.random.seed(0)

CSV = "dataset_trajectoires.csv"
N_TRAJ = None          # nb de trajectoires utilisées (None = toutes ; ex. 5000 pour aller plus vite)
EPOCHS, PATIENCE = 100, 10
LR = 3e-3
IDX_PLOT = [0, 1]      # indices (dans le jeu de test) des trajectoires à tracer


# ---------------------------------------------------------------- 3. DONNÉES
df = pd.read_csv(CSV)
groupes = {i: g.sort_values("N") for i, g in df.groupby("traj_id")}   # 1 DataFrame par trajectoire
T_MAX = max(len(g) for g in groupes.values())                          # 83 images max

ids = np.array(list(groupes.keys()))
np.random.RandomState(0).shuffle(ids)
ids_tous = ids.copy()                 # toutes les trajectoires (pour l'extraction)
if N_TRAJ: ids = ids[:N_TRAJ]         # celles utilisées pour entraîner / valider / tester

def decouper(ids):
    n = len(ids); n_train, n_val = int(0.70 * n), int(0.15 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

def calculer_stats(ids_train):            # normalisation : stats du train uniquement
    px = np.concatenate([groupes[i][["x_px", "y_px"]].values for i in ids_train])
    m  = np.concatenate([groupes[i][["x_m", "y_m", "z_m"]].values for i in ids_train])
    return px.mean(0), px.std(0), m.mean(0), m.std(0)

def trajectoires(ids, stats):
    mu_px, sd_px, mu_m, sd_m = stats
    X = np.zeros((len(ids), T_MAX, 2), dtype=np.float32)   # entrées (N, T_MAX, 2), complétées par des 0
    Y = np.zeros((len(ids), T_MAX, 3), dtype=np.float32)   # cibles  (N, T_MAX, 3)
    L = np.zeros(len(ids), dtype=np.int64)                 # vraie longueur de chaque trajectoire
    for k, i in enumerate(ids):
        g = groupes[i]; T = len(g)
        X[k, :T] = (g[["x_px", "y_px"]].values - mu_px) / sd_px
        Y[k, :T] = (g[["x_m", "y_m", "z_m"]].values - mu_m) / sd_m
        L[k] = T
    return torch.tensor(X), torch.tensor(Y), torch.tensor(L)

def preparer(batch_size=32):
    i_tr, i_va, i_te = decouper(ids)
    stats = calculer_stats(i_tr)
    loaders = []
    for part, shuffle in ((i_tr, True), (i_va, False), (i_te, False)):
        loaders.append(DataLoader(TensorDataset(*trajectoires(part, stats)),
                                  batch_size=batch_size if shuffle else 128, shuffle=shuffle))
    return (*loaders, stats)


# ---------------------------------------------------------------- 4. MODÈLE
class Reconstruction3D(nn.Module):
    """LSTM bidirectionnel, 2 couches, 64 unités cachées : (batch, T, 2) -> (batch, T, 3)."""
    def __init__(self, n_features=2, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=n_layers,
                            batch_first=True, bidirectional=True)
        self.tete = nn.Linear(hidden * 2, 3)

    def forward(self, x, lengths):                       # x : (batch, T_MAX, 2)
        T = x.shape[1]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        sorties, _ = self.lstm(packed)
        sorties, _ = pad_packed_sequence(sorties, batch_first=True, total_length=T)
        return self.tete(sorties)                        # (batch, T_MAX, 3)


# ---------------------------------------------------------------- 5. ENTRAÎNEMENT
def perte_masquee(pred, y, lengths):
    masque = (torch.arange(y.shape[1])[None, :] < lengths[:, None]).unsqueeze(-1)   # (batch, T_MAX, 1)
    return (((pred - y) ** 2) * masque).sum() / (masque.sum() * y.shape[2])

def evaluer(model, loader):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb, lb in loader:
            total += perte_masquee(model(xb, lb), yb, lb).item() * lb.sum().item()
            n += lb.sum().item()
    return total / n

def entrainer(model, train_loader, val_loader, epochs=100, lr=3e-3, patience=10, clip=1.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    meilleur, attente, meilleur_etat = float("inf"), 0, None

    for epoch in range(epochs):
        model.train()
        somme, n = 0.0, 0
        for xb, yb, lb in train_loader:
            optim.zero_grad()
            perte = perte_masquee(model(xb, lb), yb, lb)        # 1) forward + 2) loss
            perte.backward()                                    # 3) gradients (BPTT)
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # anti-explosion
            optim.step()                                        # 4) mise à jour
            somme += perte.item() * lb.sum().item(); n += lb.sum().item()

        l_train, l_val = somme / n, evaluer(model, val_loader)
        if l_val < meilleur:
            meilleur, attente, meilleur_etat = l_val, 0, copy.deepcopy(model.state_dict())
        else:
            attente += 1
            if attente >= patience:
                print(f"Arrêt anticipé à l'epoch {epoch}")
                break
        print(f"epoch {epoch:3d} | train {l_train:.4f} | val {l_val:.4f}", flush=True)

    model.load_state_dict(meilleur_etat)                 # on garde le meilleur modèle (sur la validation)

def erreur_cm(model, loader, stats):                     # RMSE en cm pour x, y, z
    sd_m = torch.tensor(stats[3], dtype=torch.float32)
    model.eval(); se, n = torch.zeros(3), 0
    with torch.no_grad():
        for xb, yb, lb in loader:
            masque = (torch.arange(yb.shape[1])[None, :] < lb[:, None]).unsqueeze(-1)
            se += ((((model(xb, lb) - yb) * sd_m) ** 2) * masque).sum((0, 1)); n += masque.sum().item()
    return ((se / n).sqrt() * 100).numpy()


# ---------------------------------------------------------------- 6. TRACÉ : réseau vs vérité
def tracer(model, test_loader, stats, indices, fichier="trajectoires_3d.png"):
    mu_m, sd_m = stats[2], stats[3]
    X, Y, L = test_loader.dataset.tensors
    model.eval()
    fig = plt.figure(figsize=(16, 4.5 * len(indices)))
    for r, k in enumerate(indices):
        T = L[k].item()
        with torch.no_grad():
            pred = model(X[k:k + 1], L[k:k + 1])[0, :T].numpy() * sd_m + mu_m   # retour en mètres
        vrai = Y[k, :T].numpy() * sd_m + mu_m
        err = np.linalg.norm(pred - vrai, axis=1).mean() * 100                    # erreur 3D moyenne (cm)

        ax = fig.add_subplot(len(indices), 4, 4 * r + 1, projection="3d")
        ax.plot(*vrai.T, "o-", color="tab:blue", ms=3, label="vraie")
        ax.plot(*pred.T, "x--", color="tab:red", ms=4, label="réseau")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(f"Test n°{k} ({T} images)\nerreur 3D moyenne : {err:.1f} cm", fontsize=10)
        ax.legend()
        for c, nom in enumerate(("x", "y", "z")):
            a = fig.add_subplot(len(indices), 4, 4 * r + 2 + c)
            a.plot(vrai[:, c], "o-", color="tab:blue", ms=3, label="vraie")
            a.plot(pred[:, c], "x--", color="tab:red", ms=4, label="réseau")
            a.set_xlabel("image"); a.set_ylabel(f"{nom}_m (m)"); a.set_title(f"{nom}_m"); a.grid(alpha=0.3)
            if c == 0: a.legend()
    fig.tight_layout(); fig.savefig(fichier, dpi=130)
    print("Figure enregistrée :", fichier)
    plt.show()


# ---------------------------------------------------------------- 6b. EXTRACTION DES TRAJECTOIRES PRÉDITES
def extraire(model, stats, choix=("test",), fichier="trajectoires_predites.csv"):
    """Un seul passage du réseau (sans gradient, sans entraînement) sur les trajectoires demandées,
    puis écriture de vrai + prédit dans un CSV.
    choix : sous-ensemble de ("train", "val", "test"), ou ("tout",) pour les 10 221 trajectoires.
    Colonne 'split' : train / val / test (comme à l'entraînement) ou 'non_utilisee' si N_TRAJ en a écarté."""
    mu_m, sd_m = stats[2], stats[3]
    i_tr, i_va, i_te = decouper(ids)
    parts = {"train": i_tr, "val": i_va, "test": i_te}
    etiquette = {i: nom for nom, p in parts.items() for i in p}
    cibles = ids_tous if "tout" in choix else np.concatenate([parts[c] for c in choix])

    blocs = []
    model.eval()
    with torch.no_grad():
        for debut in range(0, len(cibles), 256):
            lot = cibles[debut:debut + 256]
            X, _, L = trajectoires(lot, stats)
            pred = model(X, L).numpy() * sd_m + mu_m                      # (lot, T_MAX, 3), en mètres
            L = L.numpy()
            pred = pred[np.arange(pred.shape[1])[None, :] < L[:, None]]   # on retire le remplissage -> (sum(L), 3)
            vrai = np.concatenate([groupes[i][["x_m", "y_m", "z_m"]].values for i in lot])
            blocs.append(pd.DataFrame({
                "traj_id": np.repeat(lot, L),
                "N": np.concatenate([groupes[i]["N"].values for i in lot]),
                "split": np.repeat([etiquette.get(i, "non_utilisee") for i in lot], L),
                "x_m": vrai[:, 0], "y_m": vrai[:, 1], "z_m": vrai[:, 2],           # vraies valeurs (CSV d'origine)
                "x_pred": pred[:, 0].round(5), "y_pred": pred[:, 1].round(5), "z_pred": pred[:, 2].round(5),
            }))
    pd.concat(blocs).to_csv(fichier, index=False)
    print(f"{len(cibles)} trajectoires extraites ({', '.join(choix)}) -> {fichier}")


# ---------------------------------------------------------------- 7. ENTRAÎNEMENT FINAL + TEST
if __name__ == "__main__":
    if "extraire" in sys.argv[1:]:                    # --- pas de réentraînement : on recharge le réseau
        choix = [a for a in sys.argv[1:] if a != "extraire"] or ["test"]
        assert all(c in ("train", "val", "test", "tout") for c in choix), "choix possibles : train, val, test, tout"
        model = Reconstruction3D()
        model.load_state_dict(torch.load("lstm_trajectoires.pt"))
        d = np.load("normalisation.npz")
        stats = (d["mu_px"], d["sd_px"], d["mu_m"], d["sd_m"])
        extraire(model, stats, choix)
    else:
        train_loader, val_loader, test_loader, stats = preparer()
        model = Reconstruction3D()

        t0 = time.time()
        entrainer(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, patience=PATIENCE)
        print(f"Temps d'entraînement : {(time.time() - t0) / 60:.1f} min")

        print("MSE test (normalisée) :", round(evaluer(model, test_loader), 5))   # le test, UNE seule fois
        print("RMSE test en cm (x, y, z) :", erreur_cm(model, test_loader, stats).round(2))

        torch.save(model.state_dict(), "lstm_trajectoires.pt")                    # uniquement les poids du réseau
        np.savez("normalisation.npz", mu_px=stats[0], sd_px=stats[1], mu_m=stats[2], sd_m=stats[3])

        extraire(model, stats)
        tracer(model, test_loader, stats, IDX_PLOT)

    # Pour recharger plus tard :
    #   model = Reconstruction3D(); model.load_state_dict(torch.load("lstm_trajectoires.pt")); model.eval()
