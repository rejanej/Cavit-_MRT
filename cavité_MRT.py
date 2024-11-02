### Réjane Joyard - Cavité entrainée chauffée MRT ###

import numpy as np
import matplotlib.pyplot as plt

#### Paramètres d'entrées ###
m = 100
tau = 1  # Pas de temps
delta = 1  # Pas d'espace
rho0 = 5  # Densité initiale
chi0 = 0  # Température initiale
v0 = 0.2  # Vitesse initiale
U0 = 1  # Température initiale sur le bord haut
nu = 0.02  # Viscosité
wm = 1/(3*nu + 1/2)  # Temps de relaxation de la vitesse de Navier Stokes
D = 0.02817  # Coefficient de diffusion
ws = 1/(3*D + 1/2)  # Temps de relaxation de la chaleur
Nb_iter = 20000  # Nombre d'itérations

### Initialisation pour la vitesse ###
rho = rho0 * np.ones((m + 1, m + 1))
f = [4 * rho/9, rho/9, rho/9, rho/9, rho/9, rho/36, rho/36, rho/36, rho/36]

### Initialisation pour la température ###
chi = np.zeros((m+1,m+1))
g = [4 * chi/9, chi/9, chi/9, chi/9, chi/9, chi/36, chi/36, chi/36, chi/36]
chi[0, :] = U0

# Matrice
M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 0, -1, 0, 1, -1, -1, 1],
              [0, 0, 1, 0, -1, 1, 1, -1, -1],
              [-4, -1, -1, -1, -1, 2, 2, 2, 2],
              [4, -2, -2, -2, -2, 1, 1, 1, 1],
              [0, -2, 0, 2, 0, 1, -1, -1, 1],
              [0, 0, -2, 0, 2, 1, 1, -1, -1],
              [0, 1, -1, 1, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype='float64')
# Matrice inverse
M_inv = np.linalg.inv(M)

def passage_moments(M, f):
    m0 = M[0][0]*f[0] + M[0][1]*f[1] + M[0][2]*f[2] + M[0][3]*f[3] + M[0][4]*f[4] + M[0][5]*f[5] + M[0][6]*f[6] + M[0][7]*f[7] + M[0][8]*f[8]
    m1 = M[1][0]*f[0] + M[1][1]*f[1] + M[1][2]*f[2] + M[1][3]*f[3] + M[1][4]*f[4] + M[1][5]*f[5] + M[1][6]*f[6] + M[1][7]*f[7] + M[1][8]*f[8]
    m2 = M[2][0]*f[0]+ M[2][1]*f[1] + M[2][2]*f[2] + M[2][3]*f[3] + M[2][4]*f[4] + M[2][5]*f[5] + M[2][6]*f[6] + M[2][7]*f[7] + M[2][8]*f[8]
    m3 = M[3][0]*f[0] + M[3][1]*f[1] + M[3][2]*f[2] + M[3][3]*f[3] + M[3][4]*f[4] + M[3][5]*f[5] + M[3][6]*f[6] + M[3][7]*f[7] + M[3][8]*f[8]
    m4 = M[4][0]*f[0] + M[4][1]*f[1] + M[4][2]*f[2] + M[4][3]*f[3] + M[4][4]*f[4] + M[4][5]*f[5] + M[4][6]*f[6] + M[4][7]*f[7] + M[4][8]*f[8]
    m5 = M[5][0]*f[0]+ M[5][1]*f[1] + M[5][2]*f[2] + M[5][3]*f[3] + M[5][4]*f[4] + M[5][5]*f[5] + M[5][6]*f[6] + M[5][7]*f[7] + M[5][8]*f[8]
    m6 = M[6][0]*f[0] + M[6][1]*f[1] + M[6][2]*f[2] + M[6][3]*f[3] + M[6][4]*f[4] + M[6][5]*f[5] + M[6][6]*f[6] + M[6][7]*f[7] + M[6][8]*f[8]
    m7 = M[7][0]*f[0] + M[7][1]*f[1] + M[7][2]*f[2] + M[7][3]*f[3] + M[7][4]*f[4] + M[7][5]*f[5] + M[7][6]*f[6] + M[7][7]*f[7] + M[7][8]*f[8]
    m8 = M[8][0]*f[0] + M[8][1]*f[1] + M[8][2]*f[2] + M[8][3]*f[3] + M[8][4]*f[4] + M[8][5]*f[5] + M[8][6]*f[6] + M[8][7]*f[7] + M[8][8]*f[8]
    return m0, m1, m2, m3, m4, m5, m6, m7, m8

def g_collision(ws, m_t_1, m_t_2, m_t_3, m_t_4, m_t_5, m_t_6, m_t_7, m_t_8, m_t_1_eq, m_t_2_eq, m_t_3_eq, m_t_4_eq, m_t_5_eq, m_t_6_eq, m_t_7_eq, m_t_8_eq):
    m_t_1 = m_t_1*(1 - ws) + ws*m_t_1_eq
    m_t_2 = m_t_2*(1 - ws) + ws*m_t_2_eq
    m_t_3 = m_t_3*(1 - ws) + ws*m_t_3_eq
    m_t_4 = m_t_4*(1 - ws) + ws*m_t_4_eq
    m_t_5 = m_t_5*(1 - ws) + ws*m_t_5_eq
    m_t_6 = m_t_6*(1 - ws) + ws*m_t_6_eq
    m_t_7 = m_t_7*(1 - ws) + ws*m_t_7_eq
    m_t_8 = m_t_8*(1 - ws) + ws*m_t_8_eq
    return m_t_1, m_t_2, m_t_3, m_t_4, m_t_5, m_t_6, m_t_7, m_t_8

def passage_var(f, m0, m1, m2, m3, m4, m5, m6, m7, m8):
    f[0] = 1 / 9 * m0 - 4 / 36 * m3 + 4 / 36 * m4
    f[1] = 1 / 9 * m0 + 1 / 6 * m1 - 1 / 36 * m3 - 2 / 36 * m4 - 2 / 12 * m5 + 1 / 4 * m7
    f[2] = 1 / 9 * m0 + 1 / 6 * m2 - 1 / 36 * m3 - 2 / 36 * m4 - 2 / 12 * m6 - 1 / 4 * m7
    f[3] = 1 / 9 * m0 - 1 / 6 * m1 - 1 / 36 * m3 - 2 / 36 * m4 + 2 / 12 * m5 + 1 / 4 * m7
    f[4] = 1 / 9 * m0 - 1 / 6 * m2 - 1 / 36 * m3 - 2 / 36 * m4 + 2 / 12 * m6 - 1 / 4 * m7
    f[5] = 1 / 9 * m0 + 1 / 6 * m1 + 1 / 6 * m2 + 2 / 36 * m3 + 1 / 36 * m4 + 1 / 12 * m5 + 1 / 12 * m6 + 1 / 4 * m8
    f[6] = 1 / 9 * m0 - 1 / 6 * m1 + 1 / 6 * m2 + 2 / 36 * m3 + 1 / 36 * m4 - 1 / 12 * m5 + 1 / 12 * m6 - 1 / 4 * m8
    f[7] = 1 / 9 * m0 - 1 / 6 * m1 - 1 / 6 * m2 + 2 / 36 * m3 + 1 / 36 * m4 - 1 / 12 * m5 - 1 / 12 * m6 + 1 / 4 * m8
    f[8] = 1 / 9 * m0 + 1 / 6 * m1 - 1 / 6 * m2 + 2 / 36 * m3 + 1 / 36 * m4 + 1 / 12 * m5 - 1 / 12 * m6 - 1 / 4 * m8
    return f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]

### Boucle principale ###
plt.figure(figsize=(15, 5))

for _ in range(Nb_iter + 1):
    #### Schéma 1 ###
    #Calcul des m_i^eq
    w1 = 1.2
    w2 = 1.2
    w3 = 1.2
    w4 = 1.2
    w5 = 1.2
    w6 = 1.2
    w7 = 1.2
    w8 = 1.2

    m_0 = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
    m_1 = f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
    m_2 = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
    m_3 = -4 * f[0] - f[1] - f[2] - f[3] - f[4] + 2 * f[5] + 2 * f[6] + 2 * f[7] + 2 * f[8]
    m_4 = 4 * f[0] - 2 * f[1] - 2 * f[2] - 2 * f[3] - 2 * f[4] + f[5] + f[6] + f[7] + f[8]
    m_5 = -2 * f[1] + 2 * f[3] + f[5] - f[6] - f[7] + f[8]
    m_6 = -2 * f[2] + 2 * f[4] + f[5] + f[6] - f[7] - f[8]
    m_7 = f[1] - f[2] + f[3] - f[4]
    m_8 = f[5] - f[6] + f[7] - f[8]

    m_3eq = -2*m_0 + 3/m_0 * (m_1**2 + m_2**2)
    m_4eq = m_0 - 3/m_0 * (m_1**2 + m_2**2)
    m_5eq = -m_1
    m_6eq = -m_2
    m_7eq = (m_1**2 - m_2**2)/m_0
    m_8eq = (m_1*m_2)/m_0

    ### Collision pour les f_k ###
    m_3 = m_3*(1 - w3) + w3*m_3eq
    m_4 = m_4*(1 - w4) + w4*m_4eq
    m_5 = m_5*(1 - w5) + w5*m_5eq
    m_6 = m_6*(1 - w6) + w6*m_6eq
    m_7 = m_7*(1 - wm) + wm*m_7eq
    m_8 = m_8*(1 - wm) + wm*m_8eq

    f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8] = passage_var(f, m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8)

    #### Streaming pour les f_k ###
    f[1][:, 1:] = f[1][:, :-1]
    f[2][:-1, :] = f[2][1:, :]
    f[3][:, :-1] = f[3][:, 1:]
    f[4][1:, :] = f[4][:-1, :]
    f[5][:-1, 1:] = f[5][1:, :-1]
    f[6][:-1, :-1] = f[6][1:, 1:]
    f[7][1:, :-1] = f[7][:-1, 1:]
    f[8][1:, 1:] = f[8][:-1, :-1]

    ### Conditions Bounce Back ###
    f[1][:, 0], f[5][:, 0], f[8][:, 0] = f[3][:, 0], f[7][:, 0], f[6][:, 0]  # Bord gauche
    f[3][:, -1], f[6][:, -1], f[7][:, -1] = f[1][:, -1], f[8][:, -1], f[5][:, -1]  # Bord droit
    f[2][-1, :], f[5][-1, :], f[6][-1, :] = f[4][-1, :], f[7][-1, :], f[8][-1, :]  # Bord bas

    ### Conditions de Zhu et He (bord du haut) ###
    rho[0, :] = f[0][0, :] + f[1][0, :] + f[3][0, :] + 2*(f[2][0, :] + f[5][0, :] + f[6][0, :])
    f[4][0, :] = f[2][0, :]
    f[7][0, :] = f[5][0, :] - rho[0, :] * v0/6
    f[8][0, :] = f[6][0, :] + rho[0, :] * v0/6

    ### Calcul de rho et v ###
    rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
    v1 = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8])/rho
    v2 = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8])/rho

    ### Schéma 2 ###
    ### Données de collisions pour g_k ###
    # Moments
    m0, m1, m2, m3, m4, m5, m6, m7, m8 = passage_moments(M, g)

    # Données d'équilibre pour les moments
    m_t_0_eq = 0
    m_t_1_eq = v1*m0
    m_t_2_eq = v2*m0
    m_t_3_eq = -2*m0 + 3*m0*(v1**2 + v2**2)
    m_t_4_eq = m0
    m_t_5_eq = v1*m0*(-1 + 3*v1**2 + 3*v2**2)
    m_t_6_eq = v2*m0*(-1 + 3*v1**2 + 3*v2**2)
    m_t_7_eq = m0*(v1**2 - v2**2)
    m_t_8_eq = m0*(v1*v2)

    # Collision des moments
    m_t_1, m_t_2, m_t_3, m_t_4, m_t_5, m_t_6, m_t_7, m_t_8 = g_collision(ws, m1, m2, m3, m4, m5, m6, m7, m8,
                                                                         m_t_1_eq, m_t_2_eq, m_t_3_eq,
                                                                         m_t_4_eq, m_t_5_eq, m_t_6_eq,
                                                                         m_t_7_eq, m_t_8_eq)

    g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8] = passage_var(g, m0, m_t_1, m_t_2, m_t_3, m_t_4, m_t_5, m_t_6, m_t_7, m_t_8)

    ### Streaming pour les g_k ###
    g[1][:,1:] = g[1][:,:-1]
    g[2][:-1,:] = g[2][1:,:]
    g[3][:,:-1] = g[3][:,1:]
    g[4][1:,:] = g[4][:-1,:]
    g[5][:-1,1:] = g[5][1:,:-1]
    g[6][:-1,:-1] = g[6][1:,1:]
    g[7][1:,:-1] = g[7][:-1,1:]
    g[8][1:,1:] = g[8][:-1,:-1]

    ### Conditions aux bords pour les g_k ###
    # Bord gauche
    for i in [0, 1, 2, 5, 8]:
        if i == 0:
            g[i][1:, 0] = 0
        elif i == 8:
            g[i][1:, 0] = -g[i-2][1:, 0]
        else:
            g[i][1:, 0] = -g[i+2][1:, 0]

    # Bord droit
    for i in [0, 2, 3, 6, 7]:
        if i == 0:
            g[i][1:, -1] = 0
        elif i == 2 or i == 6:
            g[i][1:, -1] = -g[i+2][1:, -1]
        else:
            g[i][1:, -1] = -g[i-2][1:, -1]

    # Bord du haut
    for i in range(0,m+1):
        g[8][0, i] = U0/36 + U0/36 - g[6][0, i]
        g[7][0, i] = U0/36 + U0/36 - g[5][0, i]
        g[4][0, i] = U0/9 + U0/9 - g[2][0, i]
        g[1][0, i] = U0/9 + U0/9 - g[3][0, i]
        chi[0, i] = U0

    # Bord du bas (Neumann)
    for i in range(1,len(g)):
        g[i][-1, :] = g[i][-2, :]

    ### Calcul de chi ###
    chi = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8]

    ### Visualisation ###
    if _ % 170 == 0: # Affichage toutes les 170 itérations
        plt.clf()

        X = np.linspace(0, m, m+1)
        Y = np.linspace(0, m, m+1)
        X, Y = np.meshgrid(X, Y)

        ### Distribution de la chaleur de la cavité ###
        plt.subplot(1, 3, 1)
        plt.imshow(chi, cmap='hot', interpolation='kaiser')
        plt.colorbar()
        plt.axis('off')
        plt.title('Distribution de la chaleur')

        ### Isothermes ###
        plt.subplot(1, 3, 2)
        plt.imshow(chi, cmap='hot', interpolation='kaiser')
        plt.colorbar()
        plt.axis('off')
        levels = np.linspace(np.min(chi), np.max(chi), 10)
        plt.contour(X, Y, chi, levels=levels, colors='k', linewidths=0.7)
        plt.title('Isothermes')

        ### Streamlines ###
        plt.subplot(1, 3, 3)
        plt.imshow(chi, cmap='hot', interpolation='kaiser')
        plt.colorbar()
        plt.axis('off')
        plt.streamplot(X, Y, v1[np.round(X).astype(int), np.round(Y).astype(int)].T,
                       -v2[np.round(X).astype(int), np.round(Y).astype(int)].T, color='orange', density=1.2, linewidth=0.7)
        plt.title('Streamlines')

        plt.suptitle('Cavité entrainée chauffée')
        plt.pause(0.01)