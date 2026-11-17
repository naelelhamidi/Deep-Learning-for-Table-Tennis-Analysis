# =====================================================================
# DATASET PING-PONG : trajectoires 3D (mètres)  <->  trajectoires 2D (pixels)
#
# Extension de Ping_pong_slider_v2 / camera_ext :
#   1) modèle de caméra "sténopé" (position, angles, focale) qui met à plat les
#      trajectoires 3D dans l'image, au format (N, x_px, y_px) comme dans point_17 ;
#   2) calibration automatique de la caméra à partir des 4 coins de la table en pixels
#      (+ les 2 extrémités du filet si on les donne) ;
#   3) génération de N trajectoires et AJOUT à un CSV (les anciennes données sont gardées) ;
#   4) visualisation facultative 2D / 3D de chaque trajectoire.
#
# Utilisation (terminal) :
#   python dataset_pingpong.py -n 500                  -> ajoute 500 trajectoires au CSV
#   python dataset_pingpong.py -n 500 --workers 4      -> idem sur 4 coeurs
#   python dataset_pingpong.py -n 5 --visu both        -> + figure 2D+3D des 5 trajectoires
#   python dataset_pingpong.py --verif                 -> affiche seulement la caméra calibrée
#   python dataset_pingpong.py --voir 0 1 2 --visu both  -> visualise des trajectoires DÉJÀ dans le CSV
# Utilisation (Jupyter) :  %run dataset_pingpong.py -n 500
#
# Repère (identique à tes codes) : origine = centre de la table, z = 0 = plateau,
#   x = long de la table, y = largeur, z = vertical (SI : m, kg, s).
# =====================================================================
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# =====================================================================
# BLOC 0 : RÉGLAGES UTILISATEUR  (tout ce qu'on peut vouloir changer est ici)
# =====================================================================
N_PAR_DEFAUT = 10                       # nombre de trajectoires ajoutées à chaque exécution
FICHIER_CSV = "dataset_trajectoires.csv"  # le fichier est créé s'il n'existe pas, complété sinon
ECRIRE_META = True                       # fichier annexe *_meta.csv (type de coup, v0, w0, instants de rebond...)
MIROIR_ALEATOIRE = True                  # 1 trajectoire sur 2 est retournée (x,y)->(-x,-y) : la balle va
                                         # alors dans les deux sens, comme dans un vrai échange (la physique
                                         # est symétrique par rotation de 180° autour de z)
WORKERS = 1                              # nb de processus (>1 : plus rapide, à utiliser en script, pas en notebook Windows)

FPS = 50.0                               # cadence de la vidéo -> pas de temps de la base : DT = 1/FPS
DT = 1.0 / FPS                           # (point_17.mp4 est à 50 images/s ; N = numéro d'image)

# ---------------------------------------------------------------------
# CAMÉRA
# ---------------------------------------------------------------------
IMAGE_L, IMAGE_H = 1280, 720             # taille de l'image en pixels (largeur, hauteur)
CX, CY = None, None                      # centre optique en px ; None = centre de l'image

# "coins"  : la caméra est CALCULÉE à partir des coins de la table en pixels (COINS_PX ci-dessous)
# "manuel" : on utilise directement CAMERA_MANUELLE
MODE_CAMERA = "coins"

# Coins de la table en pixels (image 1280x720). Valeurs = médiane de point_17_table_net_smooth.csv.
# Correspondance avec le repère 3D (caméra de côté, comme dans la vidéo) :
#   coin1 = loin/droite (+Lt/2,+lt/2)   coin2 = près/droite (+Lt/2,-lt/2)
#   coin3 = près/gauche (-Lt/2,-lt/2)   coin4 = loin/gauche (-Lt/2,+lt/2)
# -> si ta caméra est ailleurs, change COINS_3D, pas les pixels.
COINS_PX = {"coin1": (806.0, 347.0), "coin2": (832.2, 443.3),
            "coin3": (442.0, 447.2), "coin4": (467.6, 350.4)}
# Extrémités du filet en pixels (haut des poteaux). Facultatif : mettre None pour ne pas s'en servir.
FILET_PX = {"filet1": (636.9, 437.6), "filet2": (636.4, 323.6)}   # filet1 = côté près, filet2 = côté loin
NET_DEBORD = 0.1525                      # le poteau dépasse la table de 15,25 cm de chaque côté

# Paramètres de la calibration que l'on veut IMPOSER (ils ne sont alors pas ajustés).
# Clés possibles : x, y, z (m) ; lacet, tangage, roulis (deg) ; focale (px).  Exemple : {"focale": 1260.0}
CAM_FIXE = {}

# Caméra utilisée si MODE_CAMERA = "manuel" (valeurs par défaut = résultat de la calibration sur point_17)
#   lacet   : rotation autour de la verticale, 0 = la caméra regarde vers +y (= azim=-90 de matplotlib)
#   tangage : inclinaison vers le bas, 0 = regard horizontal            (~ elev de matplotlib)
#   roulis  : rotation autour de l'axe optique
#   focale  : en pixels (FOV horizontal = 2*atan(IMAGE_L/2/focale))
CAMERA_MANUELLE = dict(x=0.033, y=-8.412, z=4.555, lacet=0.05, tangage=26.92, roulis=-0.60, focale=1265.0)

# ---------------------------------------------------------------------
# VISUALISATION (facultative)
# ---------------------------------------------------------------------
VISU = "aucune"                          # "aucune" | "2d" | "3d" | "both"
VISU_MAX = 5                             # nombre max de trajectoires visualisées (évite 500 fenêtres)
VISU_DOSSIER = None                      # ex "figures" : sauvegarde des PNG au lieu d'ouvrir des fenêtres

# =====================================================================
# BLOC 1 : CONSTANTES PHYSIQUES (identiques à camera_ext)
# =====================================================================
ht, hf = 0.76, 0.1525
lt, Lt = 1.525, 2.74
rb, m = 0.02, 2.7e-3
Cx, rho, g = 0.47, 1.2, 9.81
k = 0.5 * rho * np.pi * rb**2

l2, h2 = 2.0, 4.0
L3, l3 = Lt / 2, lt
E_REB, GRIP = 0.90, 1.0

rng = np.random.default_rng()

COINS_3D = {"coin1": (+Lt/2, +lt/2, 0.0), "coin2": (+Lt/2, -lt/2, 0.0),
            "coin3": (-Lt/2, -lt/2, 0.0), "coin4": (-Lt/2, +lt/2, 0.0)}
FILET_3D = {"filet1": (0.0, -lt/2 - NET_DEBORD, hf), "filet2": (0.0, +lt/2 + NET_DEBORD, hf)}


# =====================================================================
# BLOC 2 : CAMÉRA (modèle sténopé) + CALIBRATION SUR LES COINS
# =====================================================================
@dataclass
class Camera:
    x: float; y: float; z: float                 # position de la caméra (m), repère table
    lacet: float; tangage: float; roulis: float  # degrés
    focale: float                                # pixels
    cx: float = IMAGE_L / 2; cy: float = IMAGE_H / 2
    largeur: int = IMAGE_L; hauteur: int = IMAGE_H

    def axes(self):
        """Renvoie (droite, haut, avant) : base orthonormée de la caméra, dans le repère table."""
        a, p, r = np.radians([self.lacet, self.tangage, self.roulis])
        avant = np.array([-np.sin(a)*np.cos(p), np.cos(a)*np.cos(p), -np.sin(p)])
        droite = np.cross(avant, [0, 0, 1.0]); droite /= np.linalg.norm(droite)
        haut = np.cross(droite, avant)
        return (np.cos(r)*droite + np.sin(r)*haut, -np.sin(r)*droite + np.cos(r)*haut, avant)

    def projeter(self, P):
        """Points 3D (..., 3) en mètres -> pixels (..., 2) (x vers la droite, y vers le bas).
        NaN pour les points situés derrière la caméra."""
        P = np.asarray(P, float)
        droite, haut, avant = self.axes()
        d = P - np.array([self.x, self.y, self.z])
        zc = d @ avant
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.focale * (d @ droite) / zc
            v = self.cy - self.focale * (d @ haut) / zc
        out = np.stack([u, v], axis=-1)
        out[zc <= 1e-3] = np.nan
        return out

    def dans_image(self, px):
        """Booléen : le pixel est dans l'image (False aussi si NaN)."""
        with np.errstate(invalid="ignore"):
            return (px[..., 0] >= 0) & (px[..., 0] < self.largeur) & (px[..., 1] >= 0) & (px[..., 1] < self.hauteur)

    def fov_horizontal(self):
        return np.degrees(2*np.arctan(self.largeur / 2 / self.focale))


NOMS_PARAM = ["x", "y", "z", "lacet", "tangage", "roulis", "focale"]


def calibrer_depuis_coins(coins_px, coins_3d, filet_px=None, filet_3d=None, fixes=None,
                          cx=None, cy=None, largeur=IMAGE_L, hauteur=IMAGE_H):
    """Trouve (position, angles, focale) tels que les points 3D se projettent sur les pixels donnés.
    Moindres carrés non linéaires, avec plusieurs points de départ (distance, tangage, focale) pour
    ne pas dépendre d'une bonne estimation initiale. `fixes` = {nom: valeur} paramètres imposés."""
    fixes = dict(fixes or {})
    cx = largeur / 2 if cx is None else cx
    cy = hauteur / 2 if cy is None else cy
    P3 = [coins_3d[n] for n in coins_px]; PX = [coins_px[n] for n in coins_px]
    if filet_px:
        P3 += [filet_3d[n] for n in filet_px]; PX += [filet_px[n] for n in filet_px]
    P3, PX = np.array(P3, float), np.array(PX, float)
    libres = [i for i, n in enumerate(NOMS_PARAM) if n not in fixes]
    if len(PX) * 2 < len(libres):
        raise ValueError("Pas assez de points pour ajuster autant de paramètres : fixe-en (CAM_FIXE).")

    def complet(q, p0):
        p = np.array(p0, float)
        for i, n in enumerate(NOMS_PARAM):
            if n in fixes: p[i] = fixes[n]
        p[libres] = q
        return p

    def residus(q, p0):
        c = Camera(*complet(q, p0), cx=cx, cy=cy, largeur=largeur, hauteur=hauteur)
        d = P3 - np.array([c.x, c.y, c.z]); zc = d @ c.axes()[2]
        px = c.projeter(P3)
        r = np.where(np.isnan(px), 1e4, px - PX)          # point derrière la caméra : très pénalisé
        r[zc <= 1e-3] = 1e4
        return r.ravel()

    meilleur = None
    for dist in (4, 8, 14):
        for tang in (10, 30, 50):
            for foc in (800, 1500, 2800):
                p0 = [0, -dist, dist*np.tan(np.radians(tang)), 0, tang, 0, foc]
                p0 = complet(np.array(p0)[libres], p0)
                try:
                    res = least_squares(residus, np.array(p0)[libres], args=(p0,))
                except Exception:
                    continue
                if meilleur is None or res.cost < meilleur[0].cost:
                    meilleur = (res, p0)
    res, p0 = meilleur
    p = complet(res.x, p0)
    cam = Camera(*p, cx=cx, cy=cy, largeur=largeur, hauteur=hauteur)
    cam.lacet = (cam.lacet + 180) % 360 - 180            # angles ramenés dans [-180, 180]
    cam.roulis = (cam.roulis + 180) % 360 - 180
    err = np.linalg.norm(cam.projeter(P3) - PX, axis=1)
    return cam, err


def construire_camera():
    """Construit la caméra selon MODE_CAMERA. Renvoie (camera, texte_rapport)."""
    cx = CX; cy = CY
    if MODE_CAMERA == "manuel":
        cam = Camera(**CAMERA_MANUELLE, cx=IMAGE_L/2 if cx is None else cx,
                     cy=IMAGE_H/2 if cy is None else cy, largeur=IMAGE_L, hauteur=IMAGE_H)
        rapport = "Caméra : mode MANUEL"
    else:
        cam, err = calibrer_depuis_coins(COINS_PX, COINS_3D, FILET_PX, FILET_3D, CAM_FIXE, cx, cy, IMAGE_L, IMAGE_H)
        pts = list(COINS_PX) + (list(FILET_PX) if FILET_PX else [])
        rapport = "Caméra : calibrée sur " + ", ".join(pts) + "\n  erreur de reprojection (px) : " + \
                  ", ".join(f"{n}={e:.2f}" for n, e in zip(pts, err))
    rapport += (f"\n  position = ({cam.x:.3f}, {cam.y:.3f}, {cam.z:.3f}) m"
                f"\n  lacet = {cam.lacet:.2f}°  tangage = {cam.tangage:.2f}°  roulis = {cam.roulis:.2f}°"
                f"\n  focale = {cam.focale:.1f} px  (FOV horizontal = {cam.fov_horizontal():.1f}°)")
    if MODE_CAMERA == "manuel":     # en mode manuel, on contrôle quand même ce que ça donne sur les coins
        e = [np.linalg.norm(cam.projeter(COINS_3D[n]) - COINS_PX[n]) for n in COINS_PX]
        rapport += "\n  écart aux coins COINS_PX (px) : " + ", ".join(f"{n}={v:.1f}" for n, v in zip(COINS_PX, e))
    return cam, rapport


# =====================================================================
# BLOC 3 : PHYSIQUE (inchangée, copiée de camera_ext)
# =====================================================================
def CL(v, w):
    return 0.0 if w < 1e-9 else 1.0 / (2.0 + v / (rb * w))


def acc(v, w):
    sp = np.linalg.norm(v)
    a = np.array([0, 0, -g]) - k * Cx * sp * v / m
    c = np.cross(w, v); n = np.linalg.norm(c)
    if n > 1e-9:
        a += k * CL(sp, np.linalg.norm(w)) * sp**2 * c / n / m
    return a


def rk4(r, v, w, dt=0.002):
    k1v = acc(v, w);            k1r = v
    k2v = acc(v + dt/2*k1v, w); k2r = v + dt/2*k1v
    k3v = acc(v + dt/2*k2v, w); k3r = v + dt/2*k2v
    k4v = acc(v + dt*k3v, w);   k4r = v + dt*k3v
    return (r + dt/6*(k1r + 2*k2r + 2*k3r + k4r), v + dt/6*(k1v + 2*k2v + 2*k3v + k4v))


def rebond(v, w):
    ux, uy = v[0] - rb*w[1], v[1] + rb*w[0]
    Jx, Jy = -0.4*GRIP*ux, -0.4*GRIP*uy
    return (np.array([v[0] + Jx, v[1] + Jy, -E_REB*v[2]]),
            w + (1.5/rb) * np.array([Jy, -Jx, 0]))


# =====================================================================
# BLOC 4 : TIRAGE DES COUPS (identique à camera_ext)
# =====================================================================
COUPS = {
    "topspin":      dict(poids=0.30, dist=(0.5, 1.5),  haut=(0.0, 0.3),   cx=(0.5, 0.9),
                         v=(9, 19),  w=(30, 100), signe=+1, side=(0, 8, 20)),
    "slice":        dict(poids=0.15, dist=(0.3, 1.0),  haut=(-0.1, 0.2),  cx=(0.5, 0.95),
                         v=(6, 11),  w=(15, 45),  signe=-1, side=(0, 10, 25)),
    "block":        dict(poids=0.10, dist=(-0.4, 0.4), haut=(0.0, 0.2),   cx=(0.2, 0.7),
                         v=(4, 8),   w=(10, 40),  signe=+1, side=(0, 6, 15)),
    "smash":        dict(poids=0.10, dist=(-0.3, 1.2), haut=(0.4, 1),     cx=(0.4, 0.85),
                         v=(18, 28), w=(10, 40),  signe=+1, side=(0, 5, 12)),
    "poussette":    dict(poids=0.15, dist=(-0.35, 0.1), haut=(0.10, 0.25), cx=(0.2, 0.7),
                         v=(3, 8),   w=(10, 35),  signe=-1, side=(0, 8, 20)),
    "topspin_side": dict(poids=0.20, dist=(0.5, 1.5),  haut=(0.0, 0.3),   cx=(0.5, 0.9),
                         v=(8, 17),  w=(30, 90),  signe=+1, side=(40, 12, 65)),
}

Y_FRAPPE = 1.0
Y_CIBLE = 0.45
FIN_APRES = 1.0
DISP_AZ = np.radians(2.0)
DISP_EL = np.radians(1.5)
MARGE_FILET = 0.05
EL_MAX = np.radians(40)


def essai(P1, v0, w, dt=0.02):
    r, v = P1.copy(), v0.copy()
    z_filet = -1.0
    for _ in range(int(3 / dt)):
        rn, vn = rk4(r, v, w, dt)
        if r[0] < 0 <= rn[0]:
            z_filet = r[2] + (-r[0]/(rn[0] - r[0])) * (rn[2] - r[2])
        if rn[2] < 0 <= r[2]:
            return z_filet, r[0] + r[2]/(r[2] - rn[2]) * (rn[0] - r[0])
        if rn[2] < -ht:
            return z_filet, rn[0]
        r, v = rn, vn
    return z_filet, r[0]


def viser(P1, v, w, az, plage_x):
    def direction(el):
        return v * np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])

    s_filet = -P1[0] / np.cos(az)
    z_voulu = hf + rb + MARGE_FILET
    bas = np.arctan2(z_voulu - P1[2], s_filet)
    for _ in range(4):
        z_filet, _ = essai(P1, direction(bas), w)
        bas += (z_voulu - z_filet) / s_filet
    haut, el_min = EL_MAX, bas

    x_min = essai(P1, direction(bas), w)[1]
    x_max = essai(P1, direction(haut), w)[1]
    lo, hi = max(x_min, plage_x[0]), min(x_max, plage_x[1])
    if lo >= hi:
        return None
    x_cible = rng.uniform(lo, hi)

    for _ in range(8):
        milieu = (bas + haut) / 2
        if essai(P1, direction(milieu), w)[1] < x_cible: bas = milieu
        else:                                               haut = milieu
    el = (bas + haut) / 2 + rng.uniform(-DISP_EL, DISP_EL)
    return max(el, el_min), x_cible


def tirer_conditions():
    poids = np.array([c["poids"] for c in COUPS.values()])
    nom = rng.choice(list(COUPS), p=poids / poids.sum())
    c = COUPS[nom]

    P1 = np.array([-Lt/2 - rng.uniform(*c["dist"]),
                   rng.uniform(-Y_FRAPPE, Y_FRAPPE),
                   rng.uniform(*c["haut"])])
    v = rng.uniform(*c["v"])
    w_trs = rng.uniform(*c["w"])

    plage_x = (c["cx"][0]*L3, c["cx"][1]*L3)
    y_cible = rng.uniform(-Y_CIBLE, Y_CIBLE)
    az = np.arctan2(y_cible - P1[1], np.mean(plage_x) - P1[0]) + rng.uniform(-DISP_AZ, DISP_AZ)

    mu, sig, lim = c["side"]
    phi = np.radians(min(abs(rng.normal(mu, sig)), lim))
    phi *= rng.choice([-1, 1])
    axe = np.array([-np.sin(az), np.cos(az), 0.0])
    w0 = 2*np.pi*w_trs * (c["signe"]*np.cos(phi)*axe + np.sin(phi)*np.array([0, 0, 1.0]))

    visee = viser(P1, v, w0, az, plage_x)
    if visee is None:
        return None
    el = visee[0]
    d = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    return nom, P1, v*d, w0


def trajectoire(P1, v0, w, dt=DT):
    """Trajectoire complète à pas de temps uniforme dt (= 1 image vidéo).
    Renvoie (positions, (P1,P2,P3), (0,t2,t3)) ou None si rejetée."""
    r, v = P1.copy(), v0.copy()
    pts = [r.copy()]
    P2 = P3 = None
    t2 = t3 = None
    for i in range(int(4 / dt)):
        t = i * dt
        rn, vn = rk4(r, v, w, dt)

        if P2 is None and r[0] < 0 <= rn[0]:
            a = -r[0] / (rn[0] - r[0])
            P2 = r + a*(rn - r); t2 = t + a*dt
            if not (hf + rb <= P2[2] <= h2 - ht and abs(P2[1]) <= l2/2):
                return None

        if P3 is None and rn[2] < 0 <= r[2]:
            if P2 is None:
                return None
            a = r[2] / (r[2] - rn[2])
            Pi, vi = rk4(r, v, w, a*dt)
            Pi[2] = 0.0
            if not (0 <= Pi[0] <= L3 and abs(Pi[1]) <= l3/2):
                return None
            P3, t3 = Pi.copy(), t + a*dt
            v, w = rebond(vi, w)
            r, v = rk4(Pi, v, w, (1 - a)*dt)
            pts.append(r.copy())
        else:
            r, v = rn, vn
            pts.append(r.copy())

        if P3 is None and (r[2] < -ht or r[0] > Lt):
            return None
        if P3 is not None:
            if r[0] >= Lt/2 + FIN_APRES: break
            if r[2] < -ht: break
            if r[2] < 0 and r[0] <= Lt/2: return None
    if P2 is None or P3 is None:
        return None
    return np.array(pts), (P1, P2, P3), (0.0, t2, t3)


# =====================================================================
# BLOC 5 : GÉNÉRATION (avec miroir facultatif) + PROJECTION
# =====================================================================
def _miroir_z180(a):
    """Rotation de 180° autour de z : (x, y, z) -> (-x, -y, z). Valable pour positions, vitesses, rotations."""
    a = np.array(a, float).copy()
    a[..., 0] *= -1; a[..., 1] *= -1
    return a


def une_trajectoire_valide(miroir):
    """Tire des conditions jusqu'à obtenir une trajectoire valide. Renvoie un dict."""
    while True:
        cond = tirer_conditions()
        if cond is None:
            continue
        nom, P1, v0, w0 = cond
        out = trajectoire(P1, v0, w0)
        if out is None:
            continue
        pts, (P1, P2, P3), (_, t2, t3) = out
        flip = bool(miroir and rng.random() < 0.5)
        if flip:
            pts, P1, P2, P3 = map(_miroir_z180, (pts, P1, P2, P3))
            v0, w0 = _miroir_z180(v0), _miroir_z180(w0)   # w : axial, mais la rotation de 180° autour de z agit pareil
        return dict(nom=nom, miroir=flip, traj=pts, P1=P1, P2=P2, P3=P3,
                    N_filet=t2 / DT, N_rebond=t3 / DT, v0=v0, w0=w0)


def _worker(args):
    n, graine, miroir = args
    global rng
    rng = np.random.default_rng(graine)
    return [une_trajectoire_valide(miroir) for _ in range(n)]


def simuler(n, workers=1, graine=None):
    """Génère n trajectoires valides (liste de dicts)."""
    global rng
    ss = np.random.SeedSequence(graine)
    if workers <= 1:
        rng = np.random.default_rng(ss.spawn(1)[0])
        return [une_trajectoire_valide(MIROIR_ALEATOIRE) for _ in range(n)]
    from multiprocessing import Pool
    parts = [n // workers + (1 if i < n % workers else 0) for i in range(workers)]
    graines = ss.spawn(workers)
    with Pool(workers) as pool:
        blocs = pool.map(_worker, [(p, s, MIROIR_ALEATOIRE) for p, s in zip(parts, graines) if p > 0])
    return [t for b in blocs for t in b]


# =====================================================================
# BLOC 6 : CSV (ajout à chaque exécution)
#   Format "long" : une ligne par image.  Colonnes :
#   traj_id | N (numéro d'image, 0 = frappe) | x_m y_m z_m (3D, mètres) | x_px y_px (2D, pixels) | visible
#   `visible` = 1 si le pixel est dans l'image. Les pixels hors champ sont conservés (utile au RNN) ;
#   NaN seulement si la balle passe derrière la caméra.
# =====================================================================
COLONNES = ["traj_id", "N", "x_m", "y_m", "z_m", "x_px", "y_px", "visible"]
COLONNES_META = ["traj_id", "coup", "miroir", "x0", "y0", "z0", "vx0", "vy0", "vz0",
                 "wx0_rad_s", "wy0_rad_s", "wz0_rad_s", "n_images", "N_filet", "N_rebond",
                 "x_rebond", "y_rebond"]


def prochain_id(fichier):
    """Lit la dernière ligne du CSV pour connaître le dernier traj_id (rapide même sur un gros fichier)."""
    if not os.path.exists(fichier) or os.path.getsize(fichier) == 0:
        return 0
    with open(fichier, "rb") as f:
        f.seek(0, os.SEEK_END)
        taille = f.tell()
        f.seek(max(0, taille - 4096))
        lignes = f.read().decode("utf-8", "ignore").strip().splitlines()
    derniere = lignes[-1].split(",")
    try:
        return int(derniere[0]) + 1
    except ValueError:
        raise RuntimeError(f"{fichier} : dernière ligne illisible, vérifie le fichier.")


def verifier_coherence(fichier, cam):
    """Le CSV ne doit contenir qu'une seule caméra/cadence : on garde un fichier .camera.json à côté."""
    chemin = os.path.splitext(fichier)[0] + ".camera.json"
    actuel = dict(asdict(cam), dt=DT)
    if os.path.exists(chemin) and os.path.exists(fichier) and os.path.getsize(fichier) > 0:
        ancien = json.load(open(chemin))
        diff = {c: (ancien.get(c), actuel[c]) for c in actuel
                if not np.isclose(ancien.get(c, np.nan), actuel[c], rtol=1e-3, atol=0.05)}
        if diff:
            raise RuntimeError("La caméra (ou le pas de temps) a changé depuis les données déjà présentes dans "
                               f"{fichier} :\n  {diff}\nUtilise un autre fichier CSV (--csv autre.csv) ou "
                               "ajoute --forcer pour mélanger quand même.")
    return chemin, actuel


def ajouter_au_csv(res, cam, fichier):
    """Projette les trajectoires et AJOUTE les lignes au CSV (crée l'en-tête si besoin)."""
    debut = prochain_id(fichier)
    blocs, metas = [], []
    for i, r in enumerate(res):
        t = r["traj"]
        px = cam.projeter(t)
        n = len(t)
        blocs.append(pd.DataFrame({
            "traj_id": debut + i, "N": np.arange(n),
            "x_m": t[:, 0], "y_m": t[:, 1], "z_m": t[:, 2],
            "x_px": px[:, 0], "y_px": px[:, 1],
            "visible": cam.dans_image(px).astype(int)}))
        metas.append([debut + i, r["nom"], int(r["miroir"]), *r["P1"], *r["v0"], *r["w0"],
                      n, r["N_filet"], r["N_rebond"], r["P3"][0], r["P3"][1]])
    df = pd.concat(blocs, ignore_index=True)[COLONNES].round({"x_m": 5, "y_m": 5, "z_m": 5, "x_px": 2, "y_px": 2})
    nouveau = not os.path.exists(fichier) or os.path.getsize(fichier) == 0
    df.to_csv(fichier, mode="a", header=nouveau, index=False)
    if ECRIRE_META:
        fmeta = os.path.splitext(fichier)[0] + "_meta.csv"
        dm = pd.DataFrame(metas, columns=COLONNES_META).round(5)
        dm.to_csv(fmeta, mode="a", header=not os.path.exists(fmeta) or os.path.getsize(fmeta) == 0, index=False)
    return debut, len(df)


def charger_depuis_csv(fichier, ids):
    """Relit des trajectoires déjà enregistrées. Renvoie une liste de dicts {id, t (n,3), px (n,2), meta}."""
    df = pd.read_csv(fichier)
    fmeta = os.path.splitext(fichier)[0] + "_meta.csv"
    meta = pd.read_csv(fmeta).set_index("traj_id") if os.path.exists(fmeta) else None
    out = []
    for i in ids:
        d = df[df.traj_id == i].sort_values("N")
        if d.empty:
            print(f"traj_id {i} introuvable dans {fichier}")
            continue
        out.append(dict(id=i, t=d[["x_m", "y_m", "z_m"]].to_numpy(), px=d[["x_px", "y_px"]].to_numpy(),
                        meta=None if meta is None or i not in meta.index else meta.loc[i]))
    return out


# =====================================================================
# BLOC 7 : VISUALISATION FACULTATIVE (3D, 2D dans l'image, ou les deux)
# =====================================================================
def _decor_3d(ax):
    ax.plot_surface(*np.meshgrid([-Lt/2, Lt/2], [-lt/2, lt/2]), np.zeros((2, 2)), color="green", alpha=0.3)
    Y, Z = np.meshgrid([-lt/2 - NET_DEBORD, lt/2 + NET_DEBORD], [0, hf])
    ax.plot_surface(np.zeros((2, 2)), Y, Z, color="gray", alpha=0.4)


def tracer_3d(ax, t, titre="", marques=None):
    _decor_3d(ax)
    ax.plot(t[:, 0], t[:, 1], t[:, 2], color="tab:blue", lw=1.5)
    sc = ax.scatter(t[:, 0], t[:, 1], t[:, 2], c=np.arange(len(t)), cmap="viridis", s=14)
    ax.scatter(*t[0], color="k", s=40, label="N = 0")
    if marques is not None:
        ax.scatter(*marques, color="red", s=40, label="rebond")
    ax.set_xlim(-3.0, Lt/2 + FIN_APRES + 0.2); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-0.3, 1.2)
    ax.set_box_aspect((5.5, 3.0, 1.5))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(titre or "Trajectoire 3D (m)"); ax.legend(fontsize=8)
    return sc


def tracer_2d(ax, px, cam, titre="", marque_rebond=None):
    """Trajectoire dans le champ de la caméra : repère image (origine en haut à gauche), en pixels."""
    ax.add_patch(plt.Rectangle((0, 0), cam.largeur, cam.hauteur, fill=False, ec="k", lw=1.5))
    coins = cam.projeter(np.array([COINS_3D[c] for c in ["coin1", "coin2", "coin3", "coin4"]]))
    ax.add_patch(plt.Polygon(coins, closed=True, fc="green", alpha=0.25, ec="darkgreen"))
    filet = cam.projeter(np.array([FILET_3D["filet1"], FILET_3D["filet2"]]))
    bas = cam.projeter(np.array([[0, -lt/2, 0], [0, lt/2, 0]]))
    ax.plot([filet[0, 0], filet[1, 0]], [filet[0, 1], filet[1, 1]], color="gray", lw=1.5)
    for i, pos in enumerate(bas):                                 # poteaux
        ax.plot([pos[0], filet[i, 0]], [pos[1], filet[i, 1]], color="gray", lw=1.5)
    ax.plot(px[:, 0], px[:, 1], color="tab:blue", lw=1)
    sc = ax.scatter(px[:, 0], px[:, 1], c=np.arange(len(px)), cmap="viridis", s=14)
    ax.scatter(*px[0], color="k", s=40, label="N = 0")
    if marque_rebond is not None and not np.isnan(marque_rebond).any():
        ax.scatter(*marque_rebond, color="red", s=40, label="rebond")
    ax.set_xlim(-0.05*cam.largeur, 1.05*cam.largeur); ax.set_ylim(1.05*cam.hauteur, -0.05*cam.hauteur)
    ax.set_aspect("equal")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title(titre or "Trajectoire 2D (px) dans l'image"); ax.legend(fontsize=8)
    return sc


def visualiser(items, cam, mode="both", max_n=5, dossier=None):
    """items : liste de dicts {id, t (n,3), px (n,2), P3 (3,) ou None}."""
    if mode == "aucune" or not items:
        return
    if dossier:
        os.makedirs(dossier, exist_ok=True)
    for it in items[:max_n]:
        nb = 2 if mode == "both" else 1
        fig = plt.figure(figsize=(7*nb if mode != "2d" else 9, 5.5))
        pos = 1
        titre = f"traj_id {it['id']}" + (f" - {it['nom']}" if it.get("nom") else "")
        p3 = it.get("P3")
        if mode in ("3d", "both"):
            ax = fig.add_subplot(1, nb, pos, projection="3d"); pos += 1
            sc = tracer_3d(ax, it["t"], titre + " (3D, m)", p3)
        if mode in ("2d", "both"):
            ax = fig.add_subplot(1, nb, pos)
            sc = tracer_2d(ax, it["px"], cam, titre + " (2D, px)",
                           None if p3 is None else cam.projeter(np.asarray(p3)))
        fig.colorbar(sc, ax=fig.axes, shrink=0.6, label="N (image)")
        if dossier:
            fig.savefig(os.path.join(dossier, f"traj_{it['id']}_{mode}.png"), dpi=110, bbox_inches="tight")
            plt.close(fig)
    if not dossier:
        plt.show()


# =====================================================================
# BLOC 8 : PROGRAMME PRINCIPAL
# =====================================================================
def main(argv=None):
    global MODE_CAMERA
    ap = argparse.ArgumentParser(description="Génère et ajoute des trajectoires 3D/2D au CSV d'entraînement.")
    ap.add_argument("-n", type=int, default=N_PAR_DEFAUT, help="nombre de trajectoires à AJOUTER")
    ap.add_argument("--csv", default=FICHIER_CSV, help="fichier CSV (créé ou complété)")
    ap.add_argument("--camera", choices=["coins", "manuel"], default=MODE_CAMERA)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--seed", type=int, default=None, help="graine (par défaut : aléatoire à chaque exécution)")
    ap.add_argument("--visu", choices=["aucune", "2d", "3d", "both"], default=VISU)
    ap.add_argument("--visu-max", type=int, default=VISU_MAX)
    ap.add_argument("--visu-dossier", default=VISU_DOSSIER)
    ap.add_argument("--verif", action="store_true", help="affiche la caméra calibrée et s'arrête")
    ap.add_argument("--voir", type=int, nargs="+", help="visualise ces traj_id déjà présents dans le CSV et s'arrête")
    ap.add_argument("--forcer", action="store_true", help="ignore le contrôle de cohérence de la caméra")
    a, _ = ap.parse_known_args(argv)      # parse_known_args : compatible Jupyter (%run)
    MODE_CAMERA = a.camera

    cam, rapport = construire_camera()
    print(rapport)
    if a.verif:
        if a.visu != "aucune":
            t = np.array([[x, y, 0.0] for x in np.linspace(-Lt/2, Lt/2, 20) for y in (-0.3, 0.3)])
            visualiser([dict(id=-1, t=t, px=cam.projeter(t), nom="grille test")], cam, "2d", 1, a.visu_dossier)
        return

    if a.voir:
        items = charger_depuis_csv(a.csv, a.voir)
        for it in items:
            m_ = it["meta"]
            it["P3"] = None if m_ is None else np.array([m_["x_rebond"], m_["y_rebond"], 0.0])
            it["nom"] = None if m_ is None else m_["coup"]
        visualiser(items, cam, a.visu if a.visu != "aucune" else "both", a.visu_max, a.visu_dossier)
        return

    try:
        chemin_cam, cam_dict = verifier_coherence(a.csv, cam) if not a.forcer else \
            (os.path.splitext(a.csv)[0] + ".camera.json", dict(asdict(cam), dt=DT))
    except RuntimeError as e:
        sys.exit(f"\nERREUR : {e}")

    print(f"\nGénération de {a.n} trajectoires (DT = {DT:.4f} s, {a.workers} processus)...")
    res = simuler(a.n, a.workers, a.seed)
    debut, nlignes = ajouter_au_csv(res, cam, a.csv)
    json.dump(cam_dict, open(chemin_cam, "w"), indent=2)

    tot = prochain_id(a.csv)
    print(f"OK : trajectoires {debut} à {tot - 1} ajoutées ({nlignes} lignes) dans {a.csv}  "
          f"-> {tot} trajectoires au total dans le fichier.")

    if a.visu != "aucune":
        items = [dict(id=debut + i, t=r["traj"], px=cam.projeter(r["traj"]), P3=r["P3"], nom=r["nom"])
                 for i, r in enumerate(res)]
        visualiser(items, cam, a.visu, a.visu_max, a.visu_dossier)


if __name__ == "__main__":
    main(sys.argv[1:])