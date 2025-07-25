import os
from functools import partial

import h5py
from tqdm.contrib.concurrent import process_map
import multiprocessing

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# You can edit the inputs to this function to set more parameters dynamically
def get_parameters(iter, abr=1641, kE=0.3, stop_threshold=0.02, treatment_interval=1, coverage=0.65, rho=0.3, max_intervention_time=2050):
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
                "c_v": 0.005,
                "delta_h_inf": 0.003,
                "seed": seed,
                "gamma_distribution": kE,
                "delta_time_days": 1,
                "run_stop_mda_workflow": True,
                "serotest_sens_spec": [0.80, 0.99],
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
    stop_threshold=0.02,
    coverage=0.65,
    rho=0.3,
    treatment_interval=1,
):
    endgame_structure = get_parameters(i, abr=abr, kE=kE, stop_threshold=stop_threshold, treatment_interval=treatment_interval, coverage=coverage, rho=rho)

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
        # This is a list of all the default outputs you can get in at each sample point
        # You can also add custom outputs, but accessing the `state` object and outputting
        # of its attributes. This would need to be stored in a separate variable and returned
        # This specific call generates outputs for the full age range
        # However mf prevalence is limited to the minimum age of skin-snipping
        add_state_to_run_data(
            state,
            # variable to output data to
            run_data=run_data,
            # don't age group the data (outputs data for 0-80)
            with_age_groups=False,
            # output the number of people
            number=True,
            # output the number of treatments applied for each round of intervention since the last output
            n_treatments=True,
            # output the achieved coverage in the age group for each round of intervention since the last output
            achieved_coverage=True,
            # output the average mf prevalence
            prevalence=True,
            # output the mean worm burden
            mean_worm_burden=True,
            # output the OAE prevalence
            prevalence_OAE=False,
            # output the mf intensity
            intensity=True,
            # output all the sequela prevalences
            with_sequela=False,
            # Output the percent of people who don't comply
            with_pnc=True,
            # If we are also going to save data again using `add_state_to_run_data`
            # in the same timestep we need this to be True. Otherwise set it to false
            saving_multiple_states=True,
            with_ov16=True,
            with_atp=True,
            with_female_worm_burden=True,
            with_blackfly_outputs=True,
            with_stop_mda_information=output_stop_mda_info,
            ov16_sens=(80, 99),
            age_range=(5,80)
        )

        add_state_to_run_data(
            state,
            run_data=run_data_age,
            # now we want to age group the data
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
            with_ov16=True,
            with_atp=False,
            with_female_worm_burden=True,
            with_blackfly_outputs=True,
            with_stop_mda_information=output_stop_mda_info,
            ov16_sens=(80, 99),
            custom_age_groups = [(0, 5), (3, 10), (5, 10), (0, 10), (5, 15), (10, 20), (20, 80)],
            saving_multiple_states=False,
        )

    return (run_data, run_data_age)


def run_model(index, num_iters, max_workers):
    # 0.2, 0.4, 0.6, 0.8 baseline mfp, 5-80
    abr_vals = [225, 225, 225, 225, 225, 225, 400, 400, 400, 400, 400, 400, 1000, 1000, 7300, 7300]
    abr_val = abr_vals[index]
    mfp_label = "20"
    if abr_val == 400:
        mfp_label = "40"
    elif abr_val == 1000:
        mfp_label = "60"
    elif abr_val == 7300:
        mfp_label = "80"


    thresholds = [0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01]
    thresh = thresholds[index]

    coverages = [0.65, 0.65, 0.65, 0.65, 0.80, 0.80, 0.65, 0.65, 0.65, 0.65, 0.80, 0.80, 0.65, 0.65, 0.65, 0.65]
    cov = coverages[index]

    rhos = [0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    rho = rhos[index]
        

    # To use parallel processing
    # We need to make a "partial" of the function that we want to run in parallel
    # In our case this is the `run_simulations` function, and we need to pass in certain
    # parameters that will be used to inform the simulation parameters
    # These can be customized to reflect the use case of your simulation
    runSimulations = partial(
        run_simulations,
        # Verbose Output
        verbose=False,
        # How often do we want to output? 1 = every year, 0.5 = every half year, etc.
        sampling_interval=1,
        # The ABR we want to initialize the model with
        abr=abr_val,
        # The kE value we want to initialize the model with
        kE=0.3,
        # The start time of the model
        start_time=1925,
        # The end time of the model
        end_time=2051,
        stop_threshold=thresh,
        coverage=cov,
        rho=rho,
        treatment_interval=1,
    )

    # Now we use process_map to call the function we defined above
    # and tell it the number of runs we want to do
    # and the number of workers we can use.
    # It will output the data for each run in a list
    datas: list[tuple[Data, Data]] = process_map(
        runSimulations, range(num_iters), max_workers=max_workers
    )

    # this is what we use to seperate the age grouped outputs
    # from the all age outputs
    data: list[Data] = [row[0] for row in datas]
    age_data: list[Data] = [row[1] for row in datas]

    # We are then going to save this data to a csv file
    write_data_to_csv(
        data,
        f"test_outputs/python_model_output/who_schematics_for_gates/{mfp_label}_pct_mfp_{abr_val}_abr_{cov}_coverage_{rho}_rho_{thresh}_serothreshold.csv",
    )

    write_data_to_csv(
        age_data,
        f"test_outputs/python_model_output/who_schematics_for_gates/age_grouped_{mfp_label}_pct_mfp_{abr_val}_abr_{cov}_coverage_{rho}_rho_{thresh}_serothreshold.csv",
    )

# this is the function that python will start execution with when run
if __name__ == "__main__":

    # How many times we want to run the model for a given set of parameters
    # Typically this value is 200
    num_iters = 500
    max_workers = multiprocessing.cpu_count() - 1
    if num_iters < max_workers:
        max_workers = num_iters
    
    index = int(os.environ['PBS_ARRAY_INDEX']) - 1
    run_model(index=index, num_iters=num_iters, max_workers=max_workers)