import copy, random, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0); np.random.seed(0); random.seed(0)

CSV = "dataset_trajectoires.csv"
N_TESTS = 2000     # trajectoires utilisées pour les tests d'hyperparamètres (rapide)
N_FINAL = None     # trajectoires pour l'entraînement final (None = toutes)

df = pd.read_csv(CSV)
groupes = {i: g.sort_values("N") for i, g in df.groupby("traj_id")}   # 1 DataFrame par trajectoire
T_MAX = max(len(g) for g in groupes.values())                          # 83 images max

ids = np.array(list(groupes.keys()))
np.random.RandomState(0).shuffle(ids)

def decouper(ids):
    n = len(ids); n_train, n_val = int(0.70 * n), int(0.15 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

def calculer_stats(ids_train):            # normalisation : stats du train uniquement
    px = np.concatenate([groupes[i][["x_px", "y_px"]].values for i in ids_train])
    m  = np.concatenate([groupes[i][["x_m", "y_m", "z_m"]].values for i in ids_train])
    return px.mean(0), px.std(0), m.mean(0), m.std(0)

def trajectoires(ids, stats):             # remplace fenetres()
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

def preparer(n_traj, batch_size=32):
    sous = ids[:n_traj] if n_traj else ids
    i_tr, i_va, i_te = decouper(sous)
    stats = calculer_stats(i_tr)
    loaders = []
    for part, shuffle in ((i_tr, True), (i_va, False), (i_te, False)):
        loaders.append(DataLoader(TensorDataset(*trajectoires(part, stats)),
                                  batch_size=batch_size if shuffle else 128, shuffle=shuffle))
    return (*loaders, stats)

class Reconstruction3D(nn.Module):
    def __init__(self, n_features=2, hidden=64, n_layers=2, dropout=0.0, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        self.tete = nn.Linear(hidden * (2 if bidirectional else 1), 3)

    def forward(self, x, lengths):                       # x : (batch, T_MAX, 2)
        T = x.shape[1]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        sorties, _ = self.lstm(packed)
        sorties, _ = pad_packed_sequence(sorties, batch_first=True, total_length=T)
        return self.tete(sorties)                        # (batch, T_MAX, 3)

class SansMemoire(nn.Module):                            # baseline : chaque point est isolé
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
    def forward(self, x, lengths):
        return self.net(x)

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

def entrainer(model, train_loader, val_loader,
              epochs=200, lr=1e-3, patience=15, clip=1.0, weight_decay=0.0, verbose=True):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    meilleur, attente, meilleur_etat = float("inf"), 0, None
    hist = {"train": [], "val": []}

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
        hist["train"].append(l_train); hist["val"].append(l_val)

        if l_val < meilleur:
            meilleur, attente, meilleur_etat = l_val, 0, copy.deepcopy(model.state_dict())
        else:
            attente += 1
            if attente >= patience:
                if verbose: print(f"Arrêt anticipé à l'epoch {epoch}")
                break
        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train {l_train:.4f} | val {l_val:.4f}")

    model.load_state_dict(meilleur_etat)
    return hist

def erreur_cm(model, loader, stats):                         # RMSE en cm pour x, y, z
    sd_m = torch.tensor(stats[3], dtype=torch.float32)
    model.eval(); se, n = torch.zeros(3), 0
    with torch.no_grad():
        for xb, yb, lb in loader:
            masque = (torch.arange(yb.shape[1])[None, :] < lb[:, None]).unsqueeze(-1)
            se += ((((model(xb, lb) - yb) * sd_m) ** 2) * masque).sum((0, 1)); n += masque.sum().item()
    return ((se / n).sqrt() * 100).numpy()

def etape_baselines():
    tr, va, te, stats = preparer(N_TESTS)
    print("Prédire la moyenne : ~1.000 (par construction)")
    m = SansMemoire(); entrainer(m, tr, va, epochs=60, verbose=False)
    print(f"MLP sans mémoire : {evaluer(m, va):.4f} | RMSE cm = {erreur_cm(m, va, stats).round(1)}")

def etape_sante():                                   # sur-apprendre UN batch : la loss doit tendre vers ~0
    tr, va, te, stats = preparer(N_TESTS)
    xb, yb, lb = next(iter(tr))
    model = Reconstruction3D(hidden=64, n_layers=2)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    for it in range(301):
        optim.zero_grad(); perte = perte_masquee(model(xb, lb), yb, lb); perte.backward(); optim.step()
        if it % 100 == 0: print(f"it {it:3d} | loss {perte.item():.5f}")

def essai(cfg, n_traj=N_TESTS, epochs=40, patience=8):
    tr, va, te, stats = preparer(n_traj, batch_size=cfg.get("batch", 32))
    m = Reconstruction3D(hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                         dropout=cfg.get("dropout", 0.0), bidirectional=cfg.get("bi", True))
    t0 = time.time()
    h = entrainer(m, tr, va, epochs=epochs, lr=cfg["lr"], patience=patience,
                  weight_decay=cfg.get("wd", 0.0), verbose=False)
    return min(h["val"]), h["train"][int(np.argmin(h["val"]))], erreur_cm(m, va, stats), time.time() - t0

def afficher(nom, r):
    print(f"{nom:34s} val {r[0]:.4f} | train {r[1]:.4f} | RMSE cm {r[2].round(1)} | {r[3]:.0f}s", flush=True)

def etape_tests():
    base = dict(hidden=64, n_layers=2, lr=3e-3)
    print("=== Sens du réseau ===")
    for bi in (False, True):
        afficher(f"bidirectionnel={bi}", essai({**base, "bi": bi}))
    print("=== Learning rate ===")
    for lr in (3e-4, 1e-3, 3e-3, 1e-2):
        afficher(f"lr={lr}", essai({**base, "lr": lr}))
    print("=== Capacité : couches x hidden ===")
    for n_layers in (1, 2, 3):
        for hidden in (32, 64, 128):
            afficher(f"couches={n_layers} hidden={hidden}", essai({**base, "n_layers": n_layers, "hidden": hidden}))

def etape_recherche_aleatoire(n_essais=8):           # même idée que dans le tutoriel, une fois la zone repérée
    resultats = []
    for _ in range(n_essais):
        cfg = {"hidden": random.choice([64, 128]), "n_layers": random.choice([2, 3]),
               "lr": 10 ** random.uniform(-3.5, -2), "dropout": random.choice([0.0, 0.1])}
        r = essai(cfg); resultats.append((r[0], cfg)); afficher(str(cfg), r)
    for score, cfg in sorted(resultats, key=lambda r: r[0])[:3]:
        print(f"{score:.4f}", cfg)

def etape_final(cfg=None):
    cfg = cfg or dict(hidden=128, n_layers=2, lr=3e-3)   # À REMPLACER par la meilleure config des tests
    tr, va, te, stats = preparer(N_FINAL)
    model = Reconstruction3D(hidden=cfg["hidden"], n_layers=cfg["n_layers"], dropout=cfg.get("dropout", 0.0))
    entrainer(model, tr, va, epochs=150, lr=cfg["lr"], patience=15)
    print("MSE test (normalisée) :", round(evaluer(model, te), 5))        # le test, UNE seule fois
    print("RMSE test en cm (x, y, z) :", erreur_cm(model, te, stats).round(2))
    torch.save({"etat": model.state_dict(), "cfg": cfg, "stats": stats}, "lstm_trajectoires.pt")
    return model

if __name__ == "__main__":
    etapes = sys.argv[1:] or ["baselines", "sante", "tests", "final"]
    if "baselines" in etapes: etape_baselines()
    if "sante" in etapes: etape_sante()
    if "tests" in etapes: etape_tests()
    if "alea" in etapes: etape_recherche_aleatoire()
    if "final" in etapes: etape_final()