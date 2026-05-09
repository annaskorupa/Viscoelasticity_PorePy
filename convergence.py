import numpy as np


errors_x_y = [7.321265e-05, 6.985469e-05, 7.097584e-05,7.218948e-05]
errors_total = [ 1.901134e-10, 1.901134e-10, 1.901134e-10, 1.901134e-10]

errors = errors_total

for i in range(len(errors)):
    if i == 0:
        # Dla pierwszej siatki nie mamy z czym porównać
        print(f"{i:<10} | {errors[i]:.2e} | {'-':<10}")
    else:
        # Obliczamy r na podstawie bieżącego i poprzedniego błędu
        r = np.log2(errors[i-1] / errors[i])
        print(f"{i:<10} | {errors[i]:.2e} | {r:.2f}")