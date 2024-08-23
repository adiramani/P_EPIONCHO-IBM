import os
from functools import partial

import h5py
import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# You can edit the inputs to this function to set more parameters dynamically
def get_parameters(iter, abr=1641, kE=0.3, coverages=[]):
    # all treatment (MDA) that you want to apply will be stored as a list of dictionaries
    # Each dictionary will describe the MDA being applied
    # If you want to apply vector control, it is considered a model change (explained below)
    treatment_program = []

    for coverage in coverages:
        treatment_program.append(coverage)

    # changes to the model parameters will also be stored as a list of dictionaries
    changes = []
    # setting the seed of the model is optional, but good practice
    seed = iter + iter * 3758

    # Finally, we return back a dictionary full of all the parameters that we need to start the model.
    # A full list of the parameters can be found in `epioncho_ibm/state/params.py`
    return {
        "parameters": {
            "initial": {
                "n_people": 400,
                "year_length_days": 365,
                "delta_h_zero": 0.186,
                "c_v": 0.005,
                "delta_h_inf": 0.003,
                "seed": seed,
                "gamma_distribution": kE,
                "delta_time_days": 1,
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
    end_time=2005,
    coverages=[]
):
    endgame_structure = get_parameters(i, abr=abr, kE=kE, coverages=coverages)

    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)

    # EndgameSimulation is a type of Simulation that allows for changes to the parameters
    endgame_sim = EndgameSimulation(
        start_time=start_time, endgame=endgame, verbose=verbose, debug=True
    )

    # Run the Simulation and store data
    run_data: Data = {}
    run_data_age: Data = {}
    run_data_age_2: Data = {}
    for state in endgame_sim.iter_run(
        end_time=end_time, sampling_interval=sampling_interval
    ):
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
            prevalence_OAE=True,
            # output the mf intensity
            intensity=True,
            # output all the sequela prevalences
            with_sequela=True,
            # Output the percent of people who don't comply
            with_pnc=True,
            # If we are also going to save data again using `add_state_to_run_data`
            # in the same timestep we need this to be True. Otherwise set it to false
            saving_multiple_states=False,
            age_range=(5,80)
        )

        # age_groups = [(0,2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 80)]
        # add_state_to_run_data(
        #     state,
        #     run_data=run_data_age,
        #     # now we want to age group the data
        #     with_age_groups=True,
        #     number=True,
        #     n_treatments=True,
        #     achieved_coverage=True,
        #     prevalence=True,
        #     mean_worm_burden=True,
        #     prevalence_OAE=True,
        #     intensity=True,
        #     with_sequela=True,
        #     with_pnc=True,
        #     custom_age_groups=age_groups,
        #     saving_multiple_states=True,
        # )
        
        # age_groups = [(5, 15), (15, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 80)]
        # add_state_to_run_data(
        #     state,
        #     run_data=run_data_age_2,
        #     # now we want to age group the data
        #     with_age_groups=True,
        #     number=True,
        #     n_treatments=True,
        #     achieved_coverage=True,
        #     prevalence=True,
        #     mean_worm_burden=True,
        #     prevalence_OAE=True,
        #     intensity=True,
        #     with_sequela=True,
        #     with_pnc=True,
        #     custom_age_groups=age_groups,
        #     # we are not going to use `add_state_to_run_data` at this timestep anymore
        #     saving_multiple_states=False,
        # )

    return (run_data, run_data_age, run_data_age_2)


# this is the function that python will start execution with when run
if __name__ == "__main__":
    mda_values = [
        [68, 79, 78, 75, 78],
        [73, 73, 76, 78, 77],
        [77, 77, 79, 80, 79],
        [76, 56, 23, 23, 48, 65],
        [37, 39, 26, 52, 62, 66],
        [77, 64, 60, 42, 67],
        [80, 82, 84, 85, 81]
    ]
    

    max_workers = 40
    index = int(os.environ['PBS_ARRAY_INDEX']) - 1
    mda_index = index // 9
    rho_index = index % 9
    coverage_values = []
    mda_vals_to_use = mda_values[mda_index]
    mda_start_year = 1920 - len(mda_vals_to_use)
    rho_values = {
        0: np.linspace(0.9, 1, 9),
        1: np.linspace(0.9, 1, 9),
        2: np.linspace(0.9, 1, 9),
        3: np.linspace(0.9, 1, 9),
        4: np.linspace(0.9, 1, 9),
        5: np.linspace(0.9, 1, 9),
        6: np.linspace(0.9, 1, 9)
    }
    never_compliant_pct_values = {
        0: 0.17,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0.06,
    }
    if not(mda_index in rho_values):
        exit()
    for mda_val in mda_vals_to_use:
        coverage_values.append({
            "first_year": mda_start_year,
            "last_year": mda_start_year,
            "interventions": {
                "treatment_interval": 1,
                "total_population_coverage": mda_val / 100,
                "correlation": rho_values[mda_index][rho_index],
                "never_compliant_pct": never_compliant_pct_values[mda_index]
            },
        })
        mda_start_year += 1
    
    print("MDA Index")
    print(mda_index)
    print("Rho")
    print(rho_values[mda_index][rho_index])

    abr_val = 1000
    # How many times we want to run the model for a given set of parameters
    # Typically this value is 200
    num_iters = 200

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
        start_time=1900,
        # The end time of the model
        end_time=1921,
        coverages=coverage_values
    )

    # Now we use process_map to call the function we defined above
    # and tell it the number of runs we want to do
    # and the number of workers we can use.
    # It will output the data for each run in a list
    datas: list[tuple[Data, Data, Data]] = process_map(
        runSimulations, range(num_iters), max_workers=max_workers
    )

    # this is what we use to seperate the age grouped outputs
    # from the all age outputs
    data: list[Data] = [row[0] for row in datas]
    age_data: list[Data] = [row[1] for row in datas]
    age_data_2: list[Data] = [row[2] for row in datas]

    # We are then going to save this data to a csv file
    write_data_to_csv(
        data,
        "test_outputs/python_model_output/template_simulation_output_nt_" + str(never_compliant_pct_values[mda_index]) + "_mda_" + str(mda_index) + "_rho_" + str(rho_values[mda_index][rho_index]) + ".csv",
    )
    # write_data_to_csv(
    #     data,
    #     "test_outputs/python_model_output/morbidity_output_" + str(abr_val) + ".csv",
    # )
    # write_data_to_csv(
    #     age_data,
    #     "test_outputs/python_model_output/morbidity_output_age-grouped_" + str(abr_val) + ".csv",
    # )
    # write_data_to_csv(
    #     age_data_2,
    #     "test_outputs/python_model_output/morbidity_output_age-grouped_2_" + str(abr_val) + ".csv",
    # )
