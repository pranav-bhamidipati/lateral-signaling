import sacred
from sacred.observers import FileStorageObserver
from SPV_simulation_logic import do_one_simulation

ex = sacred.Experiment("SPV")
ex.observers.append(FileStorageObserver("data"))  ## We want to store data as files in the folder data

## Think of this as your default configuration
## Every variable defined in this function is handled magically by the provenance system
@ex.config
def cfg():
    p0      = 3.85
    Waa     = 0.32
    Wbb     = 0.32
    Wab     = 0.08
    kappa_A = 0.2
    kappa_P = 0.1
    v0      = 1.0e-3
    Dr      = 1e-2
    a       = 0.3
    k       = 2.
    n_c     = 100
    pE      = 0.5
    dt      = 0.01
    tfin    = 150
    save_frames = [0, -1]


@ex.automain  ## This tells python to use ex as our provenance system and to call this function as the main function
def run_one_simulation(_config, _run, seed):
    """Simulates SPV given a single parameter configuration"""
    # _config contains all the variables you define in cfg()
    # _run contains data about the run
    do_one_simulation(
        seed=seed,
        p0=_config["p0"],
        Waa=_config["Waa"],
        Wbb=_config["Wbb"],
        Wab=_config["Wab"],
        kappa_A=_config["kappa_A"],
        kappa_P=_config["kappa_P"],
        v0=_config["v0"],
        Dr=_config["Dr"],
        a=_config["a"],
        k=_config["k"],
        n_c=_config["n_c"],
        pE=_config["pE"],
        dt=_config["dt"],
        tfin=_config["tfin"],
        save=True,
        ex=ex,  ## Pass over the experiment handler ex
    )
