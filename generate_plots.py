import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("_output", exist_ok=True)

def compute_gl_weights(alpha, n):
    w = np.zeros(n + 1)
    w[0] = 1.0
    for k in range(1, n + 1):
        w[k] = w[k-1] * (1.0 - (alpha + 1.0) / k)
    return w

def run_creep_paper(
    t_end_hours=1.1,
    dt_hours=0.002,
    sigma_0=12.0,    # MPa
    sigma_s=9.0,     # MPa (yield stress)
    E1=10.91e3,      # MPa
    E2=2.76e3,       # MPa
    eta1=0.52e3,     # MPa * h^alpha
    eta20=8.21e7,    # MPa * h^beta  (8.21e4 GPa = 8.21e7 MPa)
    alpha=0.88,
    beta=0.22,
    omega=12.85,     # h^-1
    zener=False,
    burgers=False
):
    n_steps = int(t_end_hours / dt_hours)
    time = np.linspace(0, t_end_hours, n_steps + 1)
    
    eps = np.zeros(n_steps + 1)
    eps_ve = np.zeros(n_steps + 1)
    eps_vp = np.zeros(n_steps + 1)
    
    w_alpha = compute_gl_weights(alpha, n_steps)
    w_beta = compute_gl_weights(beta, n_steps)
    
    dt_alpha = dt_hours ** alpha
    dt_beta = dt_hours ** beta
    
    for n in range(1, n_steps + 1):
        t = time[n]
        
        gl_alpha = np.sum(w_alpha[1:n+1] * eps_ve[n-1::-1])
        
        # Visco-elastic branch (Fractional Kelvin-Voigt)
        if zener or burgers:
            # Zener/Burgers: alpha=1 (classical Kelvin-Voigt)
            dt_1 = dt_hours
            gl_1 = eps_ve[n-1] * -1.0 # D^1 is (u_n - u_n-1)/dt
            eps_ve[n] = (sigma_0 - (eta1/dt_1)*gl_1) / (E2 + eta1/dt_1)
        else:
            eps_ve[n] = (sigma_0 - (eta1/dt_alpha)*gl_alpha) / (E2 + eta1/dt_alpha)
            
        # Visco-plastic branch
        if sigma_0 > sigma_s and not zener:
            if burgers:
                # Burgers: beta=1, omega=0
                dt_1 = dt_hours
                gl_1 = eps_vp[n-1] * -1.0
                eps_vp[n] = dt_1 * ((sigma_0 - sigma_s) / 2.4e4) - gl_1 # Burgers eta20 = 24 GPa = 2.4e4 MPa
            else:
                gl_beta = np.sum(w_beta[1:n+1] * eps_vp[n-1::-1])
                eps_vp[n] = dt_beta * ((sigma_0 - sigma_s) / eta20) * np.exp(omega * t) - gl_beta
        else:
            eps_vp[n] = 0.0
            
        # Total strain
        eps_e = sigma_0 / E1
        eps[n] = eps_e + eps_ve[n] + eps_vp[n]
        
    return time, eps

def plot_and_save(filename, datasets, labels, colors, markers, xlabel, ylabel, title, xlim=None, ylim=None):
    plt.figure(figsize=(8, 6))
    for data, label, color, marker in zip(datasets, labels, colors, markers):
        t, eps = data
        plt.plot(t, eps*100, color=color, marker=marker, markevery=max(1, len(t)//20), linestyle='-', linewidth=1.5, label=label)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc='best', framealpha=1.0, edgecolor='black', fancybox=False, fontsize=12)
    plt.grid(False)
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(width=1.5, direction='in', top=True, right=True, labelsize=12)
    
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(f"_output/{filename}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating updated plots based on Kang et al. (2015)...")
    
    base_params = {
        't_end_hours': 1.1,
        'dt_hours': 0.002,
        'E1': 10.91e3,
        'E2': 2.76e3,
        'eta1': 0.52e3,     
        'eta20': 8.21e7,   
        'alpha': 0.88,
        'beta': 0.22,
        'omega': 12.85,
        'sigma_0': 12.0,
        'sigma_s': 9.0
    }
    
    # 1. Varying sigma_0
    print("Plot 1: Varying sigma_0")
    t1, e1 = run_creep_paper(**{**base_params, 'sigma_0': 3.0, 'omega': 0.0, 't_end_hours': 6.0})
    t2, e2 = run_creep_paper(**{**base_params, 'sigma_0': 6.0, 'omega': 0.0, 't_end_hours': 6.0})
    t3, e3 = run_creep_paper(**{**base_params, 'sigma_0': 12.0, 't_end_hours': 1.1})
    plot_and_save("plot_1_sigma", [(t1, e1), (t2, e2), (t3, e3)], 
                  [r"Experimental data with $\sigma_0 = 3.0 MPa$", r"Experimental data with $\sigma_0 = 6.0 MPa$", r"Experimental data with $\sigma_0 = 12.0 MPa$"], 
                  ['blue', 'red', 'green'], ['x', 's', 'o'], 
                  r"$t$ (h)", r"$\epsilon$ (%)", "", ylim=(0, 1.2), xlim=(0, 6))

    # 2. vs Zener
    print("Plot 2: vs Zener")
    t1, e1 = run_creep_paper(**{**base_params, 'zener': True, 't_end_hours': 8.0, 'sigma_0': 3.0}) 
    t2, e2 = run_creep_paper(**{**base_params, 't_end_hours': 8.0, 'sigma_0': 3.0})
    plot_and_save("plot_2_zener", [(t2, e2), (t1, e1)], 
                  [r"Simulation with present model", r"Simulation with classical Zener Model"], 
                  ['black', 'purple'], ['None', 'None'], 
                  r"$t$ (h)", r"$\epsilon$ (%)", "", xlim=(0, 8.0))

    # 3. vs Burgers
    print("Plot 3: vs Burgers")
    t1, e1 = run_creep_paper(**{**base_params, 'burgers': True, 't_end_hours': 1.1, 'E1': 12.0e3, 'E2': 2.68e3, 'eta1': 0.54e3})
    t2, e2 = run_creep_paper(**{**base_params, 't_end_hours': 1.1})
    plot_and_save("plot_3_burgers", [(t2, e2), (t1, e1)], 
                  [r"Simulation with present model", r"Simulation with classical Burgers model"], 
                  ['black', 'blue'], ['None', 'None'], 
                  r"$t$ (h)", r"$\epsilon$ (%)", "", xlim=(0.1, 1.05))

    # 4. Varying alpha
    print("Plot 4: Varying alpha")
    t1, e1 = run_creep_paper(**{**base_params, 'alpha': 0.65})
    t2, e2 = run_creep_paper(**{**base_params, 'alpha': 0.80})
    t3, e3 = run_creep_paper(**{**base_params, 'alpha': 0.95})
    plot_and_save("plot_4_alpha", [(t1, e1), (t2, e2), (t3, e3)], 
                  [r"$\alpha = 0.65$", r"$\alpha = 0.80$", r"$\alpha = 0.95$"], 
                  ['red', 'blue', 'black'], ['*', 'x', 'o'], 
                  r"$t$ (h)", r"$\epsilon$ (%)", "", xlim=(0.1, 1.1))

    # 5. Varying beta
    print("Plot 5: Varying beta")
    t1, e1 = run_creep_paper(**{**base_params, 'beta': 0.2})
    t2, e2 = run_creep_paper(**{**base_params, 'beta': 0.5})
    t3, e3 = run_creep_paper(**{**base_params, 'beta': 0.8})
    plot_and_save("plot_5_beta", [(t1, e1), (t2, e2), (t3, e3)], 
                  [r"$\beta = 0.2$", r"$\beta = 0.5$", r"$\beta = 0.8$"], 
                  ['black', 'blue', 'red'], ['o', 'x', '*'], 
                  r"$t$ (h)", r"$\epsilon$ (%)", "", xlim=(0.1, 1.1))

    # 6. Varying omega
    print("Plot 6: Varying omega")
    t1, e1 = run_creep_paper(**{**base_params, 'omega': 12.0, 't_end_hours': 1.0})
    t2, e2 = run_creep_paper(**{**base_params, 'omega': 6.0, 't_end_hours': 2.0})
    t3, e3 = run_creep_paper(**{**base_params, 'omega': 3.0, 't_end_hours': 4.0})
    plot_and_save("plot_6_omega", [(t1, e1), (t2, e2), (t3, e3)], 
                  [r"$\omega = 12.0 h^{-1}$", r"$\omega = 6.0 h^{-1}$", r"$\omega = 3.0 h^{-1}$"], 
                  ['black', 'blue', 'red'], ['o', 'x', '*'], 
                  r"$t$ (h)", r"$\epsilon$ (%)", "", xlim=(0, 4.0))

    # 7. Varying eta1
    print("Plot 7: Varying eta1")
    t1, e1 = run_creep_paper(**{**base_params, 'eta1': 0.52e3})
    t2, e2 = run_creep_paper(**{**base_params, 'eta1': 0.69e3})
    t3, e3 = run_creep_paper(**{**base_params, 'eta1': 0.92e3})
    plot_and_save("plot_7_eta1", [(t1, e1), (t2, e2), (t3, e3)], 
                  [r"$\eta_1 = 0.52 GPa\cdot h^\alpha$", r"$\eta_1 = 0.69 GPa\cdot h^\alpha$", r"$\eta_1 = 0.92 GPa\cdot h^\alpha$"], 
                  ['black', 'blue', 'red'], ['o', 'x', '*'], 
                  r"$t$ (h)", r"$\epsilon$ (%)", "", xlim=(0, 1.1))

    # 8. Varying eta20
    print("Plot 8: Varying eta20")
    t1, e1 = run_creep_paper(**{**base_params, 'eta20': 8.21e7})
    t2, e2 = run_creep_paper(**{**base_params, 'eta20': 8.21e8})
    t3, e3 = run_creep_paper(**{**base_params, 'eta20': 8.21e9})
    plot_and_save("plot_8_eta20", [(t1, e1), (t2, e2), (t3, e3)], 
                  [r"$\eta_{20} = 8.21\times10^4 GPa\cdot h^\beta$", r"$\eta_{20} = 8.21\times10^5 GPa\cdot h^\beta$", r"$\eta_{20} = 8.21\times10^6 GPa\cdot h^\beta$"], 
                  ['black', 'blue', 'red'], ['o', 'x', '*'], 
                  r"$t$ (h)", r"$\epsilon$ (%)", "", xlim=(0.1, 1.1))

    print("Plots generated successfully in _output/ directory.")
