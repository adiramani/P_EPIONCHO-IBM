import os
from functools import partial

import h5py
import numpy as np
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# You can edit the inputs to this function to set more parameters dynamically
def get_parameters(iter, abr=1641, kE=0.3, start_age = 5, end_age = 80):
    # all treatment (MDA) that you want to apply will be stored as a list of dictionaries
    # Each dictionary will describe the MDA being applied
    # If you want to apply vector control, it is considered a model change (explained below)
    treatment_program = []

    # changes to the model parameters will also be stored as a list of dictionaries
    changes = []
    # setting the seed of the model is optional, but good practice
    seed = iter + iter * 3758

    deltahzero = 0.186
    deltahinf = 0.003
    ch = 0.005
    if kE == 0.4:
        deltahzero = 0.118
        deltahinf = 0.002
        ch = 0.004

    # Finally, we return back a dictionary full of all the parameters that we need to start the model.
    # A full list of the parameters can be found in `epioncho_ibm/state/params.py`
    return {
        "parameters": {
            "initial": {
                "n_people": 400,
                "year_length_days": 365,
                "delta_h_zero": deltahzero,
                "c_h": ch,
                "delta_h_inf": deltahinf,
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
    start_age = 5,
    end_age = 80,
):
    endgame_structure = get_parameters(i, abr=abr, kE=kE, start_age=start_age, end_age=end_age)

    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)

    # EndgameSimulation is a type of Simulation that allows for changes to the parameters
    endgame_sim = EndgameSimulation(
        start_time=start_time, endgame=endgame, verbose=verbose, debug=True
    )

    # Run the Simulation and store data
    run_data: Data = {}
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
            age_range=(start_age, end_age)
        )

    return run_data


# this is the function that python will start execution with when run
if __name__ == "__main__":
    abr_values = [
        300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1500, 2000, 3000, 4000,
        5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 
        20000
    ]

    #36

    max_workers = 40
    index = int(os.environ['PBS_ARRAY_INDEX']) - 1

    kE_val = 0.4
    abr_value = abr_values[index]
    start_age_val = 5
    end_age_val = 80
    if index >= 35:#len(abr_values)/2:
        # start_age_val = 1
        # end_age_val = 80
        start_age_val = 5
        end_age_val = 66
    
    print("abr value")
    print(abr_values)
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
        abr=abr_value,
        # The kE value we want to initialize the model with
        kE=kE_val,
        # The start time of the model
        start_time=1900,
        # The end time of the model
        end_time=2000,
        start_age=start_age_val,
        end_age=end_age_val
    )

    # Now we use process_map to call the function we defined above
    # and tell it the number of runs we want to do
    # and the number of workers we can use.
    # It will output the data for each run in a list
    data: list[Data] = process_map(
        runSimulations, range(num_iters), max_workers=max_workers
    )

    # We are then going to save this data to a csv file
    write_data_to_csv(
        data,
        f"test_outputs/python_model_output/abr_test_outputs/test_abr_simulation_output_ke_{kE_val}_abr_{abr_value}_age_range_{start_age_val}_{end_age_val}.csv",
    )
