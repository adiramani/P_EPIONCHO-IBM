from functools import partial

from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv

# Steps to test issue
# Make sure this line https://github.com/adiramani/P_EPIONCHO-IBM/blob/testing-changes/pyproject.toml#L15C1-L15C86
# is endgame-simulations = {git = "git@github.com:dreamingspires/endgame-simulations.git"}
# poetry lock
# poetry install
# poetry run examples/test_changes.py
# You can see that in 2026, no MDA is run even though it should be, and the step taken is bigger than the delta time

# Steps to test fix
# Make sure this line https://github.com/adiramani/P_EPIONCHO-IBM/blob/testing-changes/pyproject.toml#L15C1-L15C86
# is endgame-simulations = {git = "git@github.com:adiramani/endgame-simulations.git"}
# poetry lock
# poetry install
# poetry run examples/test_changes.py
# You can see that now in 2026, MDA is run in the first time step, and the
# step taken is the same as the delta time


# Function to add model parameters (seed, exp, abr) and MDA history to endgame object
def get_endgame(iter):
    treatment_program = []
    changes = []
    seed = iter + iter * 3758
    gamma_distribution = 0.31
    abr = 600
    treatment_program.append(
        {
            "first_year": 2020,
            "last_year": 2025,
            "interventions": {
                "treatment_interval": 1,
                "total_population_coverage": 0.65,
                "correlation": 0.5,
            },
        }
    )
    treatment_program.append(
        {
            "first_year": 2026,
            "last_year": 2028,
            "interventions": {
                "treatment_interval": 1,
                "total_population_coverage": 0.65,
                "min_age_of_treatment": 4,
                "correlation": 0.5,
                "microfilaricidal_nu": 0.04,
                "microfilaricidal_omega": 1.82,
                "embryostatic_lambda_max": 462,
                "embryostatic_phi": 4.83,
            },
        }
    )
    changes.append({"year": 2026, "params": {"delta_time_days": 0.5}})

    return {
        "parameters": {
            "initial": {
                "n_people": 400,
                "year_length_days": 366,
                "delta_h_zero": 0.186,
                "c_v": 0.005,
                "delta_h_inf": 0.003,
                "seed": seed,
                "gamma_distribution": gamma_distribution,
                "delta_time_days": 1,
                "blackfly": {"bite_rate_per_person_per_year": abr},
            },
            "changes": changes,
        },
        "programs": treatment_program,
    }


# Function to run and save simulations
def run_sim(i, verbose=False):
    endgame_structure = get_endgame(i)
    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)
    endgame_sim = EndgameSimulation(
        start_time=2015, endgame=endgame, verbose=verbose, debug=True
    )
    # Run
    run_data: Data = {}
    for state in endgame_sim.iter_run(end_time=2029, sampling_interval=1):

        add_state_to_run_data(
            state,
            run_data=run_data,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=False,
            prevalence=True,
            mean_worm_burden=False,
            prevalence_OAE=False,
            intensity=False,
            with_sequela=False,
        )

    return run_data


# Wrapper
def wrapped_parameters(iu_name):
    # Run simulations and save output
    num_iter = 1
    max_workers = 1
    rumSim = partial(run_sim, verbose=False)
    data = process_map(rumSim, range(num_iter), max_workers=max_workers)
    write_data_to_csv(data, "test.csv")


if __name__ == "__main__":
    # Run example

    wrapped_parameters("test")
