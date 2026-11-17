import numpy as np
import matplotlib.pyplot as plt


# CONSTANTES PHYSIQUES ET GÉOMÉTRIQUES (SI : m, kg, s)
# Repère : origine = centre de la table, à la hauteur du plateau (z = 0 = plateau)
#   x : long de la table (x < 0 = côté joueur), y : largeur, z : vertical

ht, hf = 0.76, 0.1525        # hauteur de la table / hauteur du filet (au-dessus du plateau)
lt, Lt = 1.525, 2.74         # largeur / longueur de la table
rb, m = 0.02, 2.7e-3         # rayon et masse de la balle
Cx, rho, g = 0.47, 1.2, 9.81 # coeff. de traînée, densité de l'air, pesanteur
k = 0.5 * rho * np.pi * rb**2   # facteur commun 1/2 * rho * S

l2, h2 = 2.0, 4.0            # zone 2 (plan du filet) : largeur et hauteur max (depuis le sol)
L3, l3 = Lt / 2, lt          # zone 3 : moitié adverse de la table (là où la balle doit rebondir)
E_REB, GRIP = 0.90, 1.0      # rebond : restitution normale / accroche (1 = pas de glissement)

DT = 0.005                   # pas de temps de la base de données (s) -> à régler sur tes données test
rng = np.random.default_rng()



# RÉSOLUTION NUMERIQUE (équation du mouvement + RK4 + rebond)  [inchangé]

def CL(v, w):
    """Coefficient de portance (loi empirique) : 1/(2 + v/(rb*w)), w en rad/s.
    Proportionnel au spin si faible, sature vers 0.5."""
    return 0.0 if w < 1e-9 else 1.0 / (2.0 + v / (rb * w))


def acc(v, w):
    """Accélération = pesanteur + traînée + force de Magnus. w en rad/s."""
    sp = np.linalg.norm(v)
    a = np.array([0, 0, -g]) - k * Cx * sp * v / m          # gravité + traînée
    c = np.cross(w, v); n = np.linalg.norm(c)
    if n > 1e-9:                                             # Magnus : direction (w x v)/|w x v|
        a += k * CL(sp, np.linalg.norm(w)) * sp**2 * c / n / m
    return a


def rk4(r, v, w, dt = 0.002):
    """Un pas de Runge-Kutta 4 sur (position, vitesse), avec un pas dt correspondant à une frame."""
    k1v = acc(v, w);           k1r = v
    k2v = acc(v + dt/2*k1v, w); k2r = v + dt/2*k1v
    k3v = acc(v + dt/2*k2v, w); k3r = v + dt/2*k2v
    k4v = acc(v + dt*k3v, w);   k4r = v + dt*k3v
    return (r + dt/6*(k1r + 2*k2r + 2*k3r + k4r), v + dt/6*(k1v + 2*k2v + 2*k3v + k4v))


def rebond(v, w):
    """Choc sur la table, sans glissement, balle creuse (I = 2/3 m rb^2).
    - vz est inversée et amortie par E_REB
    - vitesse tangentielle du point de contact annulée à hauteur de GRIP
      (impulsion tangentielle = 0.4 * u, où 0.4 vient de I et m)
    - la même impulsion modifie le spin (le topspin accélère la balle, le slice la freine)."""
    ux, uy = v[0] - rb*w[1], v[1] + rb*w[0]       # vitesse du point de contact
    Jx, Jy = -0.4*GRIP*ux, -0.4*GRIP*uy
    return (np.array([v[0] + Jx, v[1] + Jy, -E_REB*v[2]]),
            w + (1.5/rb) * np.array([Jy, -Jx, 0]))



# TYPES DE COUPS (plages de valeurs plausibles en match)
#
#   poids : probabilité de choisir ce coup (normalisée automatiquement)
#   dist  : distance derrière le bord de la table (m) ; négatif = balle frappée au-dessus de la table
#   haut  : hauteur de frappe au-dessus du plateau (m)
#   cx    : point visé, en fraction de la demi-table adverse (0 = filet, 1 = bord)
#   v     : vitesse (m/s)         w : rotation totale (tr/s)
#   signe : +1 topspin, -1 backspin
#   side  : (moyenne, écart-type, limite) du SIDESPIN, en degrés. C'est l'angle phi dont l'axe de
#           rotation est incliné vers la verticale (0 = spin pur, 90 = tout en sidespin).
#           Tirage : |phi| ~ gaussienne(moyenne, écart-type), tronquée à `limite`, signe aléatoire
#           (sidespin gauche ou droit). Pour un sidespin faible : moyenne 0 et petit écart-type.

COUPS = {
    "topspin":      dict(poids=0.30, dist=(0.5, 1.5),  haut=(0.0, 0.3),   cx=(0.5, 0.9),
                         v=(9, 19),  w=(30, 100), signe=+1, side=(0, 8, 20)),
    "slice":        dict(poids=0.15, dist=(0.3, 1.0),  haut=(-0.1, 0.2),  cx=(0.5, 0.95),
                         v=(6, 11),  w=(15, 45),  signe=-1, side=(0, 10, 25)),
    "block":        dict(poids=0.10, dist=(-0.4, 0.4), haut=(0.0, 0.2),   cx=(0.2, 0.7),
                         v=(4, 8),   w=(10, 40),  signe=+1, side=(0, 6, 15)),
    "smash":        dict(poids=0.10, dist=(-0.3, 1.2), haut=(0.4, 1),     cx=(0.4, 0.85),
                         v=(18, 28), w=(10, 40),  signe=+1, side=(0, 5, 12)),
    # poussette : balle courte, frappée juste au-dessus/au bord de la table, lente, backspin léger
    "poussette":    dict(poids=0.15, dist=(-0.35, 0.1), haut=(0.10, 0.25), cx=(0.2, 0.7),
                         v=(3, 8),   w=(10, 35),  signe=-1, side=(0, 8, 20)),
    # topspin "banane" : même profil que le topspin mais l'axe est très incliné -> fort sidespin
    "topspin_side": dict(poids=0.20, dist=(0.5, 1.5),  haut=(0.0, 0.3),   cx=(0.5, 0.9),
                         v=(8, 17),  w=(30, 90),  signe=+1, side=(40, 12, 65)),
}

# Paramètres communs à tous les coups
Y_FRAPPE = 1.0                # le joueur se tient à y dans [-1, 1] m
Y_CIBLE = 0.45                # point visé à y dans [-0.45, 0.45] m (marge pour le sidespin)
FIN_APRES = 1.0               # la trajectoire continue jusqu'à 1 m après le bout de la table (zone 1')
DISP_AZ = np.radians(2.0)     # dispersion de la direction autour de la visée (azimut), "demi-cône"
DISP_EL = np.radians(1.5)     # dispersion autour de la visée (élévation)
MARGE_FILET = 0.05            # on vise au moins 5 cm au-dessus du filet
EL_MAX = np.radians(40)       # élévation maximale envisagée


# VISÉE (évite de tirer au hasard puis de rejeter presque tout)
# On choisit d'abord où la balle doit retomber, puis on cherche l'élévation qui y mène (gravité + traînée + Magnus compris) par dichotomie.

def essai(P1, v0, w, dt=0.02):
    """Intègre la balle (sans rebond ni test de validité) jusqu'à ce qu'elle redescende sous z = 0.
    Renvoie (z_filet, x_impact) : hauteur au passage du plan du filet (x = 0) et x de l'impact.
    Sert uniquement à viser. Si le filet n'est jamais atteint, z_filet = -1."""
    r, v = P1.copy(), v0.copy()
    z_filet = -1.0
    for _ in range(int(3 / dt)):
        rn, vn = rk4(r, v, w, dt)
        if r[0] < 0 <= rn[0]:                          # passage du plan du filet : interpolation
            z_filet = r[2] + (-r[0]/(rn[0] - r[0])) * (rn[2] - r[2])
        if rn[2] < 0 <= r[2]:                          # passage de z>0 à z<0 : interpolation
            return z_filet, r[0] + r[2]/(r[2] - rn[2]) * (rn[0] - r[0])
        if rn[2] < -ht:                                # tombée au sol sans avoir touché la table
            return z_filet, rn[0]
        r, v = rn, vn
    return z_filet, r[0]


def viser(P1, v, w, az, plage_x):
    """Renvoie (el, x_cible) : l'élévation (rad) et le point de chute choisi, tels que la balle
    lancée à la vitesse v, dans la direction d'azimut az, avec la rotation w, passe au-dessus du
    filet puis retombe en x = x_cible. Renvoie None si aucun point de plage_x n'est atteignable.
    Principe :
      - 'bas'  = élévation la plus plate qui passe encore MARGE_FILET au-dessus du filet
                 (plus bas, la balle accroche le filet), trouvée par 4 corrections successives ;
      - 'haut' = EL_MAX ;
      - on en déduit les x de chute atteignables [x(bas), x(haut)], on en tire un dans plage_x ;
      - entre bas et haut la distance de chute croît avec l'élévation : dichotomie."""
    def direction(el):
        return v * np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])

    s_filet = -P1[0] / np.cos(az)                      # distance horizontale P1 -> plan du filet
    z_voulu = hf + rb + MARGE_FILET
    bas = np.arctan2(z_voulu - P1[2], s_filet)         # 1re estimation : droite P1 -> dessus du filet
    for _ in range(4):                                 # la balle "s'affaisse" sous la droite : on corrige
        z_filet, _ = essai(P1, direction(bas), w)
        bas += (z_voulu - z_filet) / s_filet
    haut, el_min = EL_MAX, bas                         # el_min : en dessous, on touche le filet

    x_min = essai(P1, direction(bas), w)[1]            # chute la plus courte possible (balle la plus plate)
    x_max = essai(P1, direction(haut), w)[1]           # chute la plus longue possible
    lo, hi = max(x_min, plage_x[0]), min(x_max, plage_x[1])
    if lo >= hi:
        return None
    x_cible = rng.uniform(lo, hi)

    for _ in range(8):                                 # dichotomie (précision ~ 1 cm)
        milieu = (bas + haut) / 2
        if essai(P1, direction(milieu), w)[1] < x_cible: bas = milieu
        else:                                               haut = milieu
    el = (bas + haut) / 2 + rng.uniform(-DISP_EL, DISP_EL)   # dispersion autour de la visée
    return max(el, el_min), x_cible                           # sans jamais repasser sous le filet


# BLOC 5 : TIRAGE DES CONDITIONS INITIALES

def tirer_conditions():
    """Tire un coup plausible et renvoie (nom, P1, v0, w0), ou None si la visée est impossible.
    P1 : point de frappe (m) ; v0 : vitesse initiale (m/s) ; w0 : rotation (rad/s)."""
    poids = np.array([c["poids"] for c in COUPS.values()])
    nom = rng.choice(list(COUPS), p=poids / poids.sum())
    c = COUPS[nom]

    # 1) point de frappe (zone 1)
    P1 = np.array([-Lt/2 - rng.uniform(*c["dist"]),
                   rng.uniform(-Y_FRAPPE, Y_FRAPPE),
                   rng.uniform(*c["haut"])])

    # 2) vitesse et rotation
    v = rng.uniform(*c["v"])
    w_trs = rng.uniform(*c["w"])

    # 3) direction latérale (azimut) : on vise un point y_cible, à la profondeur moyenne du coup,
    #    + dispersion. La profondeur réelle est choisie ensuite dans viser().
    plage_x = (c["cx"][0]*L3, c["cx"][1]*L3)
    y_cible = rng.uniform(-Y_CIBLE, Y_CIBLE)
    az = np.arctan2(y_cible - P1[1], np.mean(plage_x) - P1[0]) + rng.uniform(-DISP_AZ, DISP_AZ)

    # 4) vecteur rotation : axe du topspin = z x (direction horizontale) = (-sin az, cos az, 0),
    #    incliné de phi vers la verticale pour le sidespin
    mu, sig, lim = c["side"]                           # sidespin propre au coup (degrés)
    phi = np.radians(min(abs(rng.normal(mu, sig)), lim))   # gaussienne tronquée
    phi *= rng.choice([-1, 1])                            # gauche ou droite
    axe = np.array([-np.sin(az), np.cos(az), 0.0])
    w0 = 2*np.pi*w_trs * (c["signe"]*np.cos(phi)*axe + np.sin(phi)*np.array([0, 0, 1.0]))

    # 5) élévation : celle qui mène la balle au-dessus du filet puis dans la plage visée
    visee = viser(P1, v, w0, az, plage_x)
    if visee is None:
        return None
    el = visee[0]
    d = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    return nom, P1, v*d, w0



# TRAJECTOIRE COMPLÈTE À PAS DE TEMPS UNIFORME
# Pas de temps : DT

def trajectoire(P1, v0, w, dt=DT):
    """Intègre la trajectoire et renvoie (positions, (P1,P2,P3), (0,t2,t3)) ou None si rejetée.
    Rejet si : filet non franchi dans la zone 2, rebond hors de la moitié adverse (zone 3),
    balle au sol, ou deuxième contact avec la table avant la fin de la table.
    Fin : la balle atteint le plan x = Lt/2 + FIN_APRES (1 m après la table) ou le sol."""
    r, v = P1.copy(), v0.copy()
    pts = [r.copy()]
    P2 = P3 = None
    t2 = t3 = None
    for i in range(int(4 / dt)):
        t = i * dt
        rn, vn = rk4(r, v, w, dt)

        if P2 is None and r[0] < 0 <= rn[0]:                   # passage du plan du filet
            a = -r[0] / (rn[0] - r[0])
            P2 = r + a*(rn - r); t2 = t + a*dt                 # interpolation linéaire
            if not (hf + rb <= P2[2] <= h2 - ht and abs(P2[1]) <= l2/2):
                return None

        if P3 is None and rn[2] < 0 <= r[2]:                   # impact avec z = 0
            if P2 is None:
                return None
            a = r[2] / (r[2] - rn[2])
            Pi, vi = rk4(r, v, w, a*dt)                        # on s'arrête pile à l'impact
            Pi[2] = 0.0
            if not (0 <= Pi[0] <= L3 and abs(Pi[1]) <= l3/2):
                return None
            P3, t3 = Pi.copy(), t + a*dt
            v, w = rebond(vi, w)                               # nouvelles v et w
            r, v = rk4(Pi, v, w, (1 - a)*dt)                   # on finit le pas : retour sur la grille
            pts.append(r.copy())
        else:
            r, v = rn, vn
            pts.append(r.copy())

        if P3 is None and (r[2] < -ht or r[0] > Lt):           # au sol / sorti sans rebond
            return None
        if P3 is not None:
            if r[0] >= Lt/2 + FIN_APRES: break                 # fin : zone 1' (1 m après la table)
            if r[2] < -ht: break                               # la balle touche le sol : fin
            if r[2] < 0 and r[0] <= Lt/2: return None          # 2e contact avec la table
    if P2 is None or P3 is None:
        return None
    return np.array(pts), (P1, P2, P3), (0.0, t2, t3)

# GÉNÉRATION DE LA BASE

def simuler(N):
    """Tire des conditions jusqu'à obtenir N trajectoires acceptées.
    Chaque élément : dict(nom, traj (n,3), t (n,), P (P1,P2,P3), tP (0,t2,t3), v0, w0)."""
    res, essais = [], 0
    while len(res) < N and essais < 200 * N:
        essais += 1
        cond = tirer_conditions()
        if cond is None:
            continue
        nom, P1, v0, w0 = cond
        out = trajectoire(P1, v0, w0)
        if out is not None:
            pts, P, tP = out
            res.append(dict(nom=nom, traj=pts, t=DT*np.arange(len(pts)),
                            P=P, tP=tP, v0=v0, w0=w0))
    print(f"{len(res)} trajectoires acceptées / {essais} tirages")
    return res



# Sauvegarde des données

def sauvegarder(res, fichier="base_trajectoires.npz"):
    """Enregistre la base dans un fichier .npz (numpy uniquement)."""
    n_max = max(len(r["traj"]) for r in res)
    pos = np.full((len(res), n_max, 3), np.nan)
    for i, r in enumerate(res):
        pos[i, :len(r["traj"])] = r["traj"]
    longueurs = np.array([len(r["traj"]) for r in res])
    cond_init = np.array([np.concatenate([r["P"][0], r["v0"], r["w0"]]) for r in res])
    np.savez(fichier, pos=pos, longueurs=longueurs, cond_init=cond_init, dt=DT)
    print(f"Base sauvegardée dans {fichier} : pos {pos.shape}")


# VISUALISATION 3D
# Flèche noire = v0 (longueur proportionnelle à la vitesse)
# Flèche magenta = w0 (longueur proportionnelle à la rotation, règle de la main droite)
# Couleur de la trajectoire = type de coup

ZOOM = 1                   # agrandissement de la scène (1 = taille par défaut ; trop grand => bords coupés)
V_AFF, W_AFF = 30.0, 100.0    # valeurs correspondant à une flèche de 0.8 m


def tracer(res, titre="Trajectoires simulées"):
    """Trace table, filet, trajectoires, points P1/P2/P3 et flèches v0/w0 dans une figure 3D."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0, 0, 1, 0.95], projection='3d')   # la scène occupe toute la figure

    ax.plot_surface(*np.meshgrid([-Lt/2, Lt/2], [-lt/2, lt/2]), np.zeros((2, 2)),
                    color='green', alpha=0.3)                         # table
    Y, Z = np.meshgrid([-lt/2 - 0.15, lt/2 + 0.15], [0, hf])
    ax.plot_surface(np.zeros((2, 2)), Y, Z, color='gray', alpha=0.4)  # filet
    # plan de fin (1 m après la table) pour voir où la balle repart après le rebond
    Yf, Zf = np.meshgrid([-lt/2, lt/2], [-0.3, 0.6])
    ax.plot_surface(np.full((2, 2), Lt/2 + FIN_APRES), Yf, Zf, color='red', alpha=0.08)

    couleur_coup = dict(zip(COUPS, ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']))
    for r in res:
        t, P1 = r['traj'], r['P'][0]
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=couleur_coup[r['nom']], lw=1.2)
        for P, c in zip(r['P'], ['k', 'gray', 'brown']):
            ax.scatter(*P, color=c, s=12)                             # P1, P2, P3
        ax.quiver(*P1, *(r['v0'] / V_AFF * 0.8), color='black', lw=1.2)
        ax.quiver(*P1, *(r['w0'] / (2*np.pi*W_AFF) * 0.8), color='magenta', lw=1.2)

    for nom, c in couleur_coup.items():
        ax.plot([], [], color=c, label=nom)
    ax.plot([], [], 'k', label='v0'); ax.plot([], [], 'm', label='w0')
    ax.legend(fontsize=8)

    # mêmes limites à chaque fois + échelle 1:1:1 (box_aspect proportionnel aux étendues)
    ax.set_xlim(-3.0, Lt/2 + FIN_APRES + 0.2); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-0.3, 1.0)
    ax.set_box_aspect((5.5, 2.2, 1.3), zoom=ZOOM)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title(titre)
    plt.show()


import csv

# Sauvegarde des données en CSV
def sauvegarder(res, fichier="base_trajectoires.csv"):
    """Enregistre la base dans un fichier .csv."""
    with open(fichier, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Création de l'en-tête (colonnes)
        writer.writerow([
            'id_trajectoire', 'type_coup', 'frame', 'temps', 
            'x', 'y', 'z', 
            'v0_x', 'v0_y', 'v0_z', 
            'w0_x', 'w0_y', 'w0_z'
        ])
        
        # Écriture des données frame par frame
        for id_traj, r in enumerate(res):
            nom = r['nom']
            v0x, v0y, v0z = r['v0']
            w0x, w0y, w0z = r['w0']
            
            for frame, (t, pos) in enumerate(zip(r['t'], r['traj'])):
                writer.writerow([
                    id_traj, nom, frame, round(t, 4),
                    pos[0], pos[1], pos[2],
                    v0x, v0y, v0z,
                    w0x, w0y, w0z
                ])
                
    print(f"Base sauvegardée dans {fichier} ({len(res)} trajectoires)")

if __name__ == "__main__":
    N = 7        # nombre de trajectoires à générer
    res = simuler(N)
    tracer(res)
    sauvegarder(res)  # Génère désormais le fichier .csv