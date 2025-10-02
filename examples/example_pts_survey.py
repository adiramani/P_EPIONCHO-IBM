from datetime import datetime
import multiprocessing
import os
from functools import partial

from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# You can edit the inputs to this function to set more parameters dynamically
def get_parameters(iter, abr=1641, kE=0.3, prestop_serothreshold=0.02, stop_serothreshold=0.01, sero_sens_spec = [0.80, 0.99], treatment_interval=1, coverage=0.65, rho=0.3, max_intervention_time=2050):
    # all treatment (MDA) that you want to apply will be stored as a list of dictionaries
    # Each dictionary will describe the MDA being applied
    # If you want to apply vector control, it is considered a model change (explained below)
    treatment_program = [
            {
                "first_year": 2000,
                "last_year": max_intervention_time,
                "interventions": {
                    "treatment_interval": treatment_interval,
                    "total_population_coverage": coverage,
                    "correlation": rho,
                },
            }
    ]
    changes = []#[{"year": 1990, "params": {"delta_time_days": 1}}]

    # setting the seed of the model is optional, but good practice
    seed = iter + iter * 3758

    # Finally, we return back a dictionary full of all the parameters that we need to start the model.
    # A full list of the parameters can be found in `epioncho_ibm/state/params.py`
    return {
        "parameters": {
            "initial": {
                "n_people": 2000,
                "year_length_days": 365,
                "delta_h_zero": 0.186,
                "c_h": 0.005,
                "delta_h_inf": 0.003,
                "seed": seed,
                "gamma_distribution": kE,
                "delta_time_days": 1,
                "run_stop_mda_workflow": True,
                "sero_prestop_survey_threshold": prestop_serothreshold,
                "sero_stop_survey_threshold": stop_serothreshold,
                "survey_serotest_sens_spec": sero_sens_spec,
                "blackfly": {
                    "bite_rate_per_person_per_year": abr,
                },
                "humans": {
                    "min_skinsnip_age": 0,
                },
                "exposure": {"Q": 1.2},
                # Having this in the parameters makes sure that sequela prevalence is calculated
                "sequela_active": [
                    "Blindness",
                    "SevereItching",
                    "RSD",
                    "APOD",
                    "CPOD",
                    "Atrophy",
                    "HangingGroin",
                    "Depigmentation",
                ],
            },
            "changes": changes,
        },
        "programs": treatment_program,
    }


# Function to run and save simulations
def run_simulations(
    i,
    verbose=False,
    sampling_interval=1,
    abr=1641,
    kE=0.3,
    start_time=1900,
    end_time=2051,
    prestop_serothreshold = 0.02,
    stop_serothreshold=0.01,
    sero_sens_spec = [0.80, 0.99],
    coverage=0.65,
    rho=0.3,
    treatment_interval=1,
):
    endgame_structure = get_parameters(
        i, abr=abr, kE=kE, prestop_serothreshold=prestop_serothreshold, sero_sens_spec=sero_sens_spec,
        stop_serothreshold=stop_serothreshold, treatment_interval=treatment_interval,
        coverage=coverage, rho=rho
    )

    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)

    # EndgameSimulation is a type of Simulation that allows for changes to the parameters
    endgame_sim = EndgameSimulation(
        start_time=start_time, endgame=endgame, verbose=verbose, debug=True
    )

    # Run the Simulation and store data
    run_data: Data = {}
    run_data_age: Data = {}
    output_stop_mda_info = False
    for state in endgame_sim.iter_run(
        end_time=end_time, sampling_interval=sampling_interval
    ):
        if (state.current_time + sampling_interval >= end_time):
            output_stop_mda_info = True
        add_state_to_run_data(
            state,
            run_data=run_data,
            with_age_groups=False,
            number=True,
            n_treatments=True,
            achieved_coverage=True,
            prevalence=True,
            mean_worm_burden=True,
            prevalence_OAE=False,
            intensity=True,
            with_sequela=False,
            with_pnc=True,
            saving_multiple_states=True,
            with_ov16 = True,
            with_blackfly_outputs = True,
            with_stop_mda_information = output_stop_mda_info,
            ov16_sens_spec = sero_sens_spec,
            age_range=(5,80)
        )

        add_state_to_run_data(
            state,
            run_data=run_data_age,
            with_age_groups=True,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            prevalence=True,
            mean_worm_burden=True,
            prevalence_OAE=False,
            intensity=True,
            with_sequela=False,
            with_pnc=False,
            with_ov16 = True,
            with_blackfly_outputs = True,
            with_stop_mda_information = output_stop_mda_info,
            ov16_sens_spec = sero_sens_spec,
            custom_age_groups = [(0, 5), (3, 10), (5, 10), (0, 10), (5, 15), (10, 20), (20, 80)],
            saving_multiple_states=False,
        )

    return (run_data, run_data_age)


def run_model(index, num_iters, max_workers):
    # 20%, 40% baseline mfp, 5-80
    abr_vals = [
        225, 400, 225, 225, 225, 225, 225, 225, 225,
        225, 400, 400, 400, 400, 400, 400, 400, 400
    ]
    # 1000 (60%), 7300 (80%)
    abr_val = abr_vals[index]
    mfp_label = "20"
    if abr_val == 400:
        mfp_label = "40"
    elif abr_val == 1000:
        mfp_label = "60"
    elif abr_val == 7300:
        mfp_label = "80"


    prestop_thresholds = [
        0.02, 0.02, 0.01, 0.02, 0.015, 0.01, 0.02, 0.015, 0.01,
        0.015, 0.015, 0.01, 0.02, 0.015, 0.01, 0.02, 0.015, 0.01
    ]
    prestop_thresh = prestop_thresholds[index]

    coverages = [
        0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.80, 0.80, 0.80, 
        0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.80, 0.80, 0.80
    ]
    cov = coverages[index]

    rhos = [
        0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 
        0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7
    ]
    rho = rhos[index]

    runSimulations = partial(
        run_simulations,
        verbose=False,
        sampling_interval=1,
        abr=abr_val,
        kE=0.3,
        start_time=1925,
        end_time=2101,
        prestop_serothreshold=prestop_thresh,
        stop_serothreshold=0.01,
        sero_sens_spec = [0.80, 0.99],
        coverage=cov,
        rho=rho,
        treatment_interval=1,
    )

    datas: list[tuple[Data, Data]] = process_map(
        runSimulations, range(num_iters), max_workers=max_workers
    )

    data: list[Data] = [row[0] for row in datas]
    age_data: list[Data] = [row[1] for row in datas]

    current_date = datetime.now().strftime("%Y%m%d")

    directory = f"test_outputs/python_model_output/example_pts_{current_date}/"
    os.makedirs(directory, exist_ok=True)

    write_data_to_csv(
        data,
        f"{directory}/{mfp_label}_pct_mfp_{abr_val}_abr_{cov}_coverage_{rho}_rho_{prestop_thresh}_serothreshold.csv",
    )

    write_data_to_csv(
        age_data,
        f"{directory}/age_grouped_{mfp_label}_pct_mfp_{abr_val}_abr_{cov}_coverage_{rho}_rho_{prestop_thresh}_serothreshold.csv",
    )

if __name__ == "__main__":
    num_iters = 500
    max_workers = multiprocessing.cpu_count() - 1
    if num_iters < max_workers:
        max_workers = num_iters
    
    index = int(os.environ.get("PBS_ARRAY_INDEX", 1)) - 1
    run_model(index=index, num_iters=num_iters, max_workers=max_workers)