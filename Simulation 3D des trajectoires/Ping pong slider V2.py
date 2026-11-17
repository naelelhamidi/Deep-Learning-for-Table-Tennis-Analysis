import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =====================================================================
# BLOC 1 : CONSTANTES (SI : m, kg, s)
# Repère : origine = centre de la table, à la hauteur du plateau (z = 0 = plateau)
#   x : long de la table (x < 0 = côté joueur), y : largeur, z : vertical
# =====================================================================
ht, hf = 0.76, 0.1525
lt, Lt = 1.525, 2.74
rb, m = 0.02, 2.7e-3
Cx, rho, g = 0.47, 1.2, 9.81
k = 0.5 * rho * np.pi * rb**2
E_REB, GRIP = 0.90, 1.0
DT = 0.005                    # pas de temps de l'intégration (s)
FIN_APRES = 1.0               # on arrête la balle 1 m après le bout de la table

# =====================================================================
# BLOC 2 : PHYSIQUE (inchangé)
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
# BLOC 3 : TRAJECTOIRE À PARTIR DES CONDITIONS INITIALES
# Intègre pas à pas ; si la balle passe sous z = 0 au-dessus de la table, rebond.
# Arrêt : sol atteint, balle à 1 m après la table, ou 3 s écoulées.
# =====================================================================
def trajectoire(P1, v0, w0):
    r, v, w = np.array(P1, float), np.array(v0, float), np.array(w0, float)
    pts = [r.copy()]
    for _ in range(int(3 / DT)):
        rn, vn = rk4(r, v, w, DT)
        sur_table = abs(rn[0]) <= Lt/2 and abs(rn[1]) <= lt/2
        if rn[2] < 0 <= r[2] and sur_table:            # impact sur la table
            a = r[2] / (r[2] - rn[2])                  # fraction du pas jusqu'à z = 0
            Pi, vi = rk4(r, v, w, a*DT)
            Pi[2] = 0.0
            pts.append(Pi.copy())
            v, w = rebond(vi, w)                       # nouvelles vitesse et rotation
            r = Pi
            continue
        r, v = rn, vn
        pts.append(r.copy())
        if r[2] < -ht or r[0] > Lt/2 + FIN_APRES: break
    return np.array(pts)



# =====================================================================
# BLOC 4 : FIGURE 3D + SLIDERS
# =====================================================================
fig = plt.figure(figsize=(13, 8))

# 1. Activation de la projection en perspective
ax = fig.add_axes([0.0, 0.30, 0.75, 0.68], projection='3d', proj_type='persp')

# 2. Ajustement de la longueur focale pour accentuer le trapèze
# Plus focal_length est petit (ex: 0.15 à 0.3), plus l'effet de fuite/perspective est fort
ax.set_proj_type('persp', focal_length=0.3)

# Vue caméra
ax.view_init(elev=22, azim=-90)
# --- Vue caméra ajustée (hauteur et angle réels) ---
# elev=25 offre le bon compromis de hauteur sans écraser la perspective
#ax.view_init(elev=25, azim=-90)

# Décor fixe : Table et filet
X_table, Y_table = np.meshgrid([-Lt/2, Lt/2], [-lt/2, lt/2])
ax.plot_surface(X_table, Y_table, np.zeros((2, 2)), color='#111111', alpha=0.35, edgecolor='#a000ff', lw=1.5, zorder=1)
ax.plot([-Lt/2, Lt/2], [0, 0], [0, 0], color='white', lw=1, zorder=2)

Y_filet, Z_filet = np.meshgrid([-lt/2 - 0.15, lt/2 + 0.15], [0, hf])
ax.plot_surface(np.zeros((2, 2)), Y_filet, Z_filet, color='#222222', alpha=0.5, edgecolor='purple', zorder=2)

# --- Cadrage et proportions strictes (empêche la déformation 3D) ---
# En gardant des amplitudes zlim et ylim équilibrées, la table conserve sa forme rééle
ax.set_xlim(-2.9, 2.9)
ax.set_ylim(-1.4, 1.4)
ax.set_zlim(-0.4, 1.4)

# Proportions naturelles des axes (Longueur, Largeur, Hauteur)
ax.set_box_aspect((5.6, 3.0, 1.8))

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')

# Trajectoire (zorder élevé pour être visible devant le plateau)
ligne, = ax.plot([], [], [], color='cyan', lw=2.5, zorder=10)
arrows = []

# Sliders
reglages = [("x0", -3.5, -1.4, -2.2), ("y0", -1.2, 1.2, 0.0), ("z0", -0.3, 1.2, 0.2),
            ("vx", 0, 30, 10),       ("vy", -10, 10, 0),      ("vz", -10, 10, 2),
            ("wx (tr/s)", -80, 80, 0), ("wy (tr/s)", -120, 120, 40), ("wz (tr/s)", -80, 80, 0)]

sliders = []
for i, (nom, lo, hi, val) in enumerate(reglages):
    col, lig = i // 3, i % 3
    axs = fig.add_axes([0.10 + 0.31*col, 0.19 - 0.055*lig, 0.18, 0.03])
    sliders.append(Slider(axs, nom, lo, hi, valinit=val))

texte = fig.text(0.78, 0.6, "", fontsize=10, va='top')

def maj(_=None):
    x0, y0, z0, vx, vy, vz, wx, wy, wz = [s.val for s in sliders]
    P1, v0 = np.array([x0, y0, z0]), np.array([vx, vy, vz])
    w0 = 2*np.pi*np.array([wx, wy, wz])

    t = trajectoire(P1, v0, w0)
    ligne.set_data_3d(t[:, 0], t[:, 1], t[:, 2])

    for a in arrows:
        a.remove()
    arrows.clear()

    arrows.append(ax.quiver(*P1, *(v0/30*0.8), color='blue', lw=2, zorder=10))
    arrows.append(ax.quiver(*P1, *(w0/(2*np.pi*80)*0.8), color='magenta', lw=2, zorder=10))

    rebonds = t[(t[:, 2] == 0)]
    texte.set_text(f"|v0| = {np.linalg.norm(v0):.1f} m/s\n|w0| = {np.linalg.norm(w0)/(2*np.pi):.1f} tr/s\n"
                   f"rebond(s) sur la table : {len(rebonds)}\nbleu = v0, magenta = w0")

    fig.canvas.draw()

for s in sliders:
    s.on_changed(maj)

maj()
plt.show()
