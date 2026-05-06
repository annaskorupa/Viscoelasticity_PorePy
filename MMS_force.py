import sympy as sp

# 1. Definicja zmiennych
x, y, t = sp.symbols('x y t')
Ax, Ay, b, beta = sp.symbols('Ax Ay b beta')
Lx, Ly, mu1, mu2, k = sp.symbols('Lx Ly mu1 mu2 k')

# 2. Definicja funkcji czasu
T1 = 1 - sp.exp(-b*t)
T2 = (b / (beta - b)) * (sp.exp(-b*t) - sp.exp(-beta*t))

# 3. Przestrzenne pole przemieszczeń
ux_s = Ax * sp.sin(sp.pi * x / Lx) * sp.cos(sp.pi * y / Ly)
uy_s = Ay * sp.sin(sp.pi * x / Lx) * sp.sin(sp.pi * y / Ly)

# 4. Odkształcenia przestrzenne
eps_xx_s = sp.diff(ux_s, x)
eps_yy_s = sp.diff(uy_s, y)
eps_xy_s = 0.5 * (sp.diff(ux_s, y) + sp.diff(uy_s, x))

# 5. Ślad i dewiator (Zakładamy Płaski Stan Odkształcenia, tr_eps = eps_xx + eps_yy)
tr_eps_s = eps_xx_s + eps_yy_s
# Pamiętaj: w 3D (i płaskim stanie) dev_xx = eps_xx - 1/3*tr_eps
dev_eps_xx_s = eps_xx_s - tr_eps_s / 3
dev_eps_yy_s = eps_yy_s - tr_eps_s / 3
dev_eps_xy_s = eps_xy_s 

# 6. Naprężenia całkowite z modelu lepkosprężystego
sig_xx = 2*mu1 * dev_eps_xx_s * T1 + 2*mu2 * dev_eps_xx_s * T2 + k * tr_eps_s * T1
sig_yy = 2*mu1 * dev_eps_yy_s * T1 + 2*mu2 * dev_eps_yy_s * T2 + k * tr_eps_s * T1
sig_xy = 2*mu1 * dev_eps_xy_s * T1 + 2*mu2 * dev_eps_xy_s * T2 # brak tr_eps w ścinaniu

# 7. Siły wymuszające MMS (f = - div(sigma))
fx = -(sp.diff(sig_xx, x) + sp.diff(sig_xy, y))
fy = -(sp.diff(sig_xy, x) + sp.diff(sig_yy, y))

# Wypisz gotowe wzory
print("--- Sila fx ---")
print(sp.simplify(fx))
print("\n--- Sila fy ---")
print(sp.simplify(fy))