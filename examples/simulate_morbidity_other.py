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

    deltahzero = 0.186
    deltahinf = 0.003
    c_h = 0.005
    if kE == 0.4:
        deltahzero = 0.118
        deltahinf = 0.002
        c_h = 0.004

    # Finally, we return back a dictionary full of all the parameters that we need to start the model.
    # A full list of the parameters can be found in `epioncho_ibm/state/params.py`
    return {
        "parameters": {
            "initial": {
                "n_people": 2000,
                "year_length_days": 365,
                "seed": seed,
                "gamma_distribution": kE,
                "delta_time_days": 1,
                "blackfly": {
                    "bite_rate_per_person_per_year": abr,
                    "delta_h_zero": deltahzero,
                    "c_h": c_h,
                    "delta_h_inf": deltahinf,
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
    coverages=[],
    start_age=1,
    end_age=80,
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
    run_data_2: Data = {}
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
            saving_multiple_states=True,
            age_range=(5,80)
        )

        add_state_to_run_data(
            state,
            run_data=run_data_age,
            # now we want to age group the data
            with_age_groups=True,
            number=True,
            n_treatments=True,
            achieved_coverage=True,
            prevalence=True,
            mean_worm_burden=True,
            prevalence_OAE=True,
            intensity=True,
            with_sequela=True,
            with_pnc=True,
            saving_multiple_states=True,
        )

        age_groups_2 = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 80)]
        add_state_to_run_data(
            state,
            run_data=run_data_age_2,
            # now we want to age group the data
            with_age_groups=True,
            number=True,
            n_treatments=True,
            achieved_coverage=True,
            prevalence=True,
            mean_worm_burden=True,
            prevalence_OAE=True,
            intensity=True,
            with_sequela=True,
            with_pnc=True,
            custom_age_groups=age_groups_2,
            saving_multiple_states=True,
        )

        add_state_to_run_data(
            state,
            # variable to output data to
            run_data=run_data_2,
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


    return (run_data, run_data_2, run_data_age, run_data_age_2)


# this is the function that python will start execution with when run
if __name__ == "__main__":
    mda_values = [
        # [76.2]*8,
        # [76.2]*17,
        # [76.2]*30,
        # [76.2]*30,
        # [72.0, 75, 75, 75, 77],
        # [72.0, 75, 75, 75, 77] + [75]*25,
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [76.2]*30,
        [76.2]*30,
        [72.0, 75, 75, 75, 77] + [75]*25,
        [72.0, 75, 75, 75, 77] + [75]*25,
    ]
    

    max_workers = 50
    index = int(os.environ['PBS_ARRAY_INDEX']) - 1
    if index == 0:
        index = 9
    elif index == 1:
        index = 10
    mda_index = index
    coverage_values = []
    mda_vals_to_use = mda_values[mda_index]
    mda_start_year = 2030 - len(mda_vals_to_use)
    never_ever_treated_values = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        # 0: 0,
        # 1: 0,
        # 2: 0,
        # 3: 0,
        # 4: 0.01,
        # 5: 0,
    }
    rho_values = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0.35,
        10: 0.65,
        11: 0,
        12: 0,
        # 0: 0.35,
        # 1: 0.65,
        # 2: 0.35,
        # 3: 0.65,
        # 4: 0,
        # 5: 0,
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
                "correlation": rho_values[mda_index],
                "never_compliant_pct": never_ever_treated_values[mda_index],
            },
        })
        mda_start_year += 1

    abr_vals = {
        0: 615,
        1: 240, # 255,
        2: 20000, #30000,
        3: 2000,
        4: 185,
        5: 285,
        6: 615,
        7: 2250,
        8: 60000,
        9: 530,# 600,
        10: 530,# 600,
        11: 40000,
        12: 40000,
    }
    abr_val = abr_vals[mda_index]
    # How many times we want to run the model for a given set of parameters
    # Typically this value is 200
    num_iters = 1000
    mda_to_site = {
        0: "Murdoch",
        1: "Teke-23",
        2: "Teke-85",
        3: "Kirkwood-68",
        4: "Kirkwood-10",
        5: "Kirkwood-30",
        6: "Kirkwood-50",
        7: "Kirkwood-70",
        8: "Kirkwood-90",
        9: "Brieger",
        10: "Osue",
        11: "Kennedy-0.3",
        12: "Kennedy-0.4"
    }

    kE_vals = {
        0: 0.3,
        1: 0.3,
        2: 0.3,
        3: 0.3,
        4: 0.3,
        5: 0.3,
        6: 0.3,
        7: 0.3,
        8: 0.3,
        9: 0.3,
        10: 0.3,
        11: 0.3,
        12: 0.4,
    }
    kE_val = kE_vals[mda_index]

    start_age_val = 1
    end_age_val = 80
    if "Kennedy" in mda_to_site[mda_index]:
        start_age_val = 5
        end_age_val = 66
    elif "Murdoch" in mda_to_site[mda_index] or "Kirkwood-68" in mda_to_site[mda_index]:
        start_age_val = 5
        end_age_val = 80

    print("MDA Index")
    print(mda_index)
    print("Rho")
    print(rho_values[mda_index])
    print("Never Ever Treated")
    print(never_ever_treated_values[mda_index])
    print("ABR")
    print(abr_val)
    print("Ke")
    print(kE_val)

    end_time_val = 2000
    if len(mda_vals_to_use) > 0:
        end_time_val = 2031

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
        kE=kE_val,
        # The start time of the model
        start_time=1900,
        # The end time of the model
        end_time=end_time_val,
        coverages=coverage_values,
        start_age=start_age_val,
        end_age=end_age_val,
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
    run_data_2: list[Data] = [row[1] for row in datas]
    age_data: list[Data] = [row[2] for row in datas]
    #age_data_2: list[Data] = [row[3] for row in datas]

    # We are then going to save this data to a csv file
    # write_data_to_csv(
    #     data,
    #     "test_outputs/python_model_output/morbidity_outputs_rerun/morbidity_output_all_ages_" + mda_to_site[mda_index] + "_abr_" + str(abr_val) + "_rho_" + str(rho_values[mda_index]) + "_nevertreated_" + str(never_ever_treated_values[mda_index]) + ".csv"
    # )
    # write_data_to_csv(
    #     run_data_2,
    #     "test_outputs/python_model_output/morbidity_outputs_rerun/morbidity_output_proper_age_range_" + mda_to_site[mda_index] + "_abr_" + str(abr_val) + "_rho_" + str(rho_values[mda_index]) + "_nevertreated_" + str(never_ever_treated_values[mda_index]) + f"_age_start_{start_age_val}_age_end_{end_age_val}" + ".csv"
    # )
    write_data_to_csv(
        age_data,
        "test_outputs/python_model_output/morbidity_output_update_1_to_3/morbidity_output_single_year_age_grouped_" + mda_to_site[mda_index] + "_abr_" + str(abr_val) + "_rho_" + str(rho_values[mda_index]) + "_nevertreated_" + str(never_ever_treated_values[mda_index]) + f"_age_start_{start_age_val}_age_end_{end_age_val}" + ".csv"
    )
    # write_data_to_csv(
    #     age_data_2,
    #     "test_outputs/python_model_output/morbidity_outputs_rerun/morbidity_output_old_age_grouped_" + mda_to_site[mda_index] + "_abr_" + str(abr_val) + "_rho_" + str(rho_values[mda_index]) + "_nevertreated_" + str(never_ever_treated_values[mda_index]) + f"_age_start_{start_age_val}_age_end_{end_age_val}" + ".csv"
    # )
