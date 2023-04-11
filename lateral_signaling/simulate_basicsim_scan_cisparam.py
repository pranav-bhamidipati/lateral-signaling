from simulate_basicsim_run_one import ex


save_frames = [0, 2000, 4000]

delta_space = [0.0, 1.0, 2.0]

for delta in delta_space:
    config_updates = {
        "delta": float(delta),
        "save_frames": save_frames,
        "animate": True,
        "n_frames": 51,
    }
    ex.run(config_updates=config_updates)  # Run with the updated parameters
