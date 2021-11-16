from SPV_run_one import ex # We import the experiment here. It knows what are the default values, how to run the code, and where to store it's results
import numpy as np

# Param values to scan
Waa_vals = np.linspace(-0.3, 0.3, 13)
Wbb = 0.0
Wab = 0.1

k   = 0.0
a   = 0.0

for Waa in Waa_vals:  # Over what parameters do we loop
    config_updates = { # Update the default variables (all others are still the same)
        "Waa": Waa,
        "Wbb": Wbb,
        "Wab": Wab,
        "k": k,
        "a": a,
    }
    ex.run(config_updates=config_updates)  # Run with the updated parameters
