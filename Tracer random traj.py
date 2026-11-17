"""
Tire au hasard 10 trajectoires dans trajectoires_predites.csv et trace vraie (bleu) vs réseau (rouge).
Aucun réseau n'est utilisé : tout vient du fichier créé par trajectoires_lstm_final.py.
Sorties : trajectoires_aleatoires_1.png et trajectoires_aleatoires_2.png (5 trajectoires chacune)
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

FICHIER = "trajectoires_predites.csv"
N_TIRAGE = 10
SPLIT = "test"      # "test" (recommandé : trajectoires jamais vues), "val", "train", ou None pour tout mélanger
GRAINE = None       # None = tirage différent à chaque exécution ; un entier (ex. 0) = tirage reproductible
PAR_FIGURE = 5      # nb de trajectoires par image


df = pd.read_csv(FICHIER)
candidats = df if SPLIT is None else df[df["split"] == SPLIT]
rng = np.random.default_rng(GRAINE)
choisies = rng.choice(candidats["traj_id"].unique(), size=N_TIRAGE, replace=False)
print("Trajectoires tirées (traj_id) :", sorted(choisies.tolist()))

groupes = {i: g.sort_values("N") for i, g in df[df["traj_id"].isin(choisies)].groupby("traj_id")}


def tracer_page(indices, fichier):
    fig = plt.figure(figsize=(16, 4.5 * len(indices)))
    for r, i in enumerate(indices):
        g = groupes[i]
        vrai = g[["x_m", "y_m", "z_m"]].values
        pred = g[["x_pred", "y_pred", "z_pred"]].values
        err = np.linalg.norm(pred - vrai, axis=1).mean() * 100            # erreur 3D moyenne (cm)

        ax = fig.add_subplot(len(indices), 4, 4 * r + 1, projection="3d")
        ax.plot(*vrai.T, "o-", color="tab:blue", ms=3, label="vraie")
        ax.plot(*pred.T, "x--", color="tab:red", ms=4, label="réseau")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(f"traj_id {i} ({len(g)} images, {g['split'].iloc[0]})\nerreur 3D moyenne : {err:.1f} cm", fontsize=10)
        ax.legend()
        for c, nom in enumerate(("x", "y", "z")):
            a = fig.add_subplot(len(indices), 4, 4 * r + 2 + c)
            a.plot(vrai[:, c], "o-", color="tab:blue", ms=3, label="vraie")
            a.plot(pred[:, c], "x--", color="tab:red", ms=4, label="réseau")
            a.set_xlabel("image"); a.set_ylabel(f"{nom}_m (m)"); a.set_title(f"{nom}_m"); a.grid(alpha=0.3)
            if c == 0: a.legend()
    fig.tight_layout(); fig.savefig(fichier, dpi=130)
    print("Figure enregistrée :", fichier)


for p, debut in enumerate(range(0, N_TIRAGE, PAR_FIGURE), start=1):
    tracer_page(choisies[debut:debut + PAR_FIGURE], f"trajectoires_aleatoires_{p}.png")
plt.show()