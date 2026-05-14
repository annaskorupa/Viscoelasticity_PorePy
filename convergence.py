import numpy as np
import matplotlib.pyplot as plt

#mesh steps: 10x10: 0.1 m, 20x10: 0.05 m, 40x40: 0.025 m, 80x80: 0.0125 m
Nx = np.array([10, 20, 40, 80])

#time steps: 8s, 4s, 2s, 1s
dt = np.array([8.0, 4.0, 2.0, 1.0])

errors_total = [ 3.276869e-11   , 8.593092e-12 , 2.181605e-12  , 5.461565e-13 ] #absolute errors [m]
errors_total_rel = np.array([1.867602e-02,4.897503e-03, 1.243373e-03, 3.112736e-04]) #relative errors [-]

errors = errors_total_rel

for i in range(len(errors)):
    if i == 0:
        # Dla pierwszej siatki nie mamy z czym porównać
        print(f"{i:<10} | {errors[i]:.2e} | {'-':<10}")
    else:
        # Obliczamy r na podstawie bieżącego i poprzedniego błędu
        r = np.log2(errors[i-1] / errors[i])
        print(f"{i:<10} | {errors[i]:.2e} | {r:.2f}")

# resolution for 2D: Nx x Nx (Nx^2) and time step dt ~1/dt - like in https://doi.org/10.5149/ARC-GR.1598 (fig. 3 and 5)
Nt = 1.0 / dt

resolution = (Nx**2 * Nt)**0.25

# plot

plt.figure(figsize=(7,5))

plt.loglog(
    resolution,
    errors_total_rel,
    marker='o',
    linewidth=2,
)

# convergence
# coeff = np.polyfit(np.log(resolution), np.log(errors_total_rel), 1)
# p = -coeff[0]


# xfit = np.linspace(resolution.min(), resolution.max(), 200)
# yfit = np.exp(coeff[1]) * xfit**coeff[0]

# plt.loglog(
#     xfit,
#     yfit,
#     linestyle='--',
#     label=f'Observed order ≈ {p:.2f}'
# )

# convergence - rectangular
x1 = resolution[-2]
x2 = resolution[-1]

y1 = errors_total_rel[-2]
y2 = errors_total_rel[-1]

offset = 1.5
yt1 = y1 * offset
yt2 = y2 * offset

plt.plot([x1, x2], [yt1, yt1], 'k-')
plt.plot([x2, x2], [yt1, yt2], 'k-')
plt.plot([x1, x2], [yt1, yt2], 'k-')
plt.text(
    np.sqrt(x1*x2),         
    np.sqrt(yt1*yt2),
    '2',
    fontsize=12
)

plt.xlabel(r'$(N_x^2 \cdot N_t)^{1/4}$')
plt.ylabel('Relative $L^2$ error')

plt.grid(True, which='both')
#plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("convergence.png", dpi=200)

print("done")
