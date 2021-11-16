import sacred
from sacred.observers import FileStorageObserver
from lsig_simulation_logic import do_one_simulation

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling")
ex.observers.append(FileStorageObserver("sacred"))  # Dir for storing results

# Read trial parameters from CSV
data_dir  = os.path.abspath("../data")
params_df = pd.read_csv(os.path.join(data_dir, "sim_parameters.csv"))

# Get arguments for dampening function (beta), delay, and interaction radius
beta_args = params_df.loc[
    params_df.parameter.str.startswith("beta_")
].value.values
delay     = params_df.loc[
    params_df.parameter.str.startswith("delay")
].value.values[0]
r_int     = params_df.loc[
    params_df.parameter.str.startswith("r_int")
].value.values[0]

# Package all other parameters 
param_names = params_df.loc[
    params_df.rhs, "parameter"
].values.astype(str)
param_vals  = params_df.loc[params_df.rhs, "value"].values

# Make tuple to pass to lsig.integrate_DDE
dde_args = tuple(param_vals)

# Get `k`
where_k = next(i for i, pn in enumerate(param_names) if "k" == pn)
k = param_vals[where_k]
thresh = k

# Get `g`
where_g = next(i for i, pn in enumerate(param_names) if "g" == pn)
g = param_vals[where_g]

# Get `lambda` (basal promoter activity)
lambda_ = dde_args[4]

# Variables here are handled magically by the provenance system
@ex.config
def cfg():
    alpha   = 3.,
    k       = 0.02,
    p       = 2.,
    delta   = 1.,
    lambda_ = 1e-3,
    delay   = 0.3,
    rep     = -1,
    g       = 1.,
    rho_0   = 1.,
    rho_max = 6.,


@ex.automain  ## This tells python to use ex as our provenance system and to call this function as the main function
def run_one_simulation(_config, _run, seed):
    """Simulates SPV given a single parameter configuration"""
    # _config contains all the variables you define in cfg()
    # _run contains data about the run
    do_one_simulation(
        alpha   = _config["alpha"],
        k       = 0.02,
        p       = 2.,
        delta   = 1.,
        lambda_ = 1e-3,
        delay   = 0.3,
        rep     = -1,
        g       = 1.,
        rho_0   = 1.,
        rho_max = 6.,
        tmax_days = 8.,
        nt_t    = 100,
        rows    = 80,
        cols    = 80,
        gamma_R = 0.1,
#         seed=seed,
#         p0=_config["p0"],
#         Waa=_config["Waa"],
#         Wbb=_config["Wbb"],
#         Wab=_config["Wab"],
#         kappa_A=_config["kappa_A"],
#         kappa_P=_config["kappa_P"],
#         v0=_config["v0"],
#         Dr=_config["Dr"],
#         a=_config["a"],
#         k=_config["k"],
#         n_c=_config["n_c"],
#         pE=_config["pE"],
#         dt=_config["dt"],
#         tfin=_config["tfin"],
        save=True,
        ex=ex,  ## Pass over the experiment handler ex
    )
