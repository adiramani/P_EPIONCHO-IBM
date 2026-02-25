import os
from functools import partial

import h5py
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# You can edit the inputs to this function to set more parameters dynamically
def get_parameters(iter, abr=1641, kE=0.3, psf=0.5):
    # all treatment (MDA) that you want to apply will be stored as a list of dictionaries
    # Each dictionary will describe the MDA being applied
    # If you want to apply vector control, it is considered a model change (explained below)
    treatment_program = []

    # changes to the model parameters will also be stored as a list of dictionaries
    changes = []
    # setting the seed of the model is optional, but good practice
    seed = iter + iter * 3758

    delta_hz_in =  0.186
    delta_hinf_in = 0.003
    c_h_in = 0.005
    if (kE == 0.2):
        delta_hz_in =  0.385
        delta_hinf_in = 0.003
        c_h_in = 0.008
    elif (kE == 0.4):
        delta_hz_in = 0.118
        delta_hinf_in = 0.002
        c_h_in = 0.004

    treatment_program.append(
        {
            "first_year": 1987,
            "last_year": 2005,
            # default treatment name is IVM
            "treatment_name": "IVM",
            "interventions": {
                # 1 = annual treatment, 0.5 = bi-annual, etc.
                "treatment_interval": 1,
                "total_population_coverage": 0.68,
                "correlation": 0.2,
            },
        }
    )
    treatment_program.append(
        {
            "first_year": 2006,
            "last_year": 2006,
            # default treatment name is IVM
            "treatment_name": "IVM",
            "interventions": {
                # 1 = annual treatment, 0.5 = bi-annual, etc.
                "treatment_interval": 1,
                "total_population_coverage": 0.702,
                "correlation": 0.2,
            },
        }
    )
    treatment_program.append(
        {
            "first_year": 2007,
            "last_year": 2007,
            # default treatment name is IVM
            "treatment_name": "IVM",
            "interventions": {
                # 1 = annual treatment, 0.5 = bi-annual, etc.
                "treatment_interval": 1,
                "total_population_coverage": 0.724,
                "correlation": 0.2,
            },
        }
    )
    treatment_program.append(
        {
            "first_year": 2008,
            "last_year": 2008,
            # default treatment name is IVM
            "treatment_name": "IVM",
            "interventions": {
                # 1 = annual treatment, 0.5 = bi-annual, etc.
                "treatment_interval": 1,
                "total_population_coverage": 0.746,
                "correlation": 0.2,
            },
        }
    )
    treatment_program.append(
        {
            "first_year": 2009,
            "last_year": 2009,
            # default treatment name is IVM
            "treatment_name": "IVM",
            "interventions": {
                # 1 = annual treatment, 0.5 = bi-annual, etc.
                "treatment_interval": 1,
                "total_population_coverage": 0.768,
                "correlation": 0.2,
            },
        }
    )
    treatment_program.append(
        {
            "first_year": 2010,
            "last_year": 2030,
            # default treatment name is IVM
            "treatment_name": "IVM",
            "interventions": {
                # 1 = annual treatment, 0.5 = bi-annual, etc.
                "treatment_interval": 0.5,
                "total_population_coverage": 0.79,
                "correlation": 0.2,
            },
        }
    )


    changes.append(
        {
            "year": 1986,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr * 0.2}},
        }
    )
    changes.append(
        {
            "year": 1987,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr}},
        }
    )
    changes.append(
        {
            "year": 1990,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr * 0.2}},
        }
    )
    changes.append(
        {
            "year": 1996,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr * 0.36}},
        }
    )
    changes.append(
        {
            "year": 1997,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr * 0.52}},
        }
    )
    changes.append(
        {
            "year": 1998,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr * 0.68}},
        }
    )
    changes.append(
        {
            "year": 1999,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr * 0.84}},
        }
    )
    changes.append(
        {
            "year": 2000,
            "params": {"blackfly": {"bite_rate_per_person_per_year": abr}},
        }
    )

    # treatment_program = []
    # changes = []

    return {
        "parameters": {
            "initial": {
                "n_people": 440,
                "year_length_days": 365,
                "seed": seed,
                "gamma_distribution": kE,
                "delta_time_days": 1,
                "blackfly": {
                    "delta_h_zero": delta_hz_in,
                    "c_h": c_h_in,
                    "delta_h_inf": delta_hinf_in,
                    "bite_rate_per_person_per_year": abr,
                },
                "humans": {
                    "min_skinsnip_age": 0,
                    "probability_serorevert_fast": psf
                },
                "exposure": {"Q": 1.2}
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
    psf = 0.5,
    start_time=1900,
    end_time=2005,
):
    endgame_structure = get_parameters(i, abr=abr, kE=kE, psf=psf)

    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)

    # EndgameSimulation is a type of Simulation that allows for changes to the parameters
    endgame_sim = EndgameSimulation(
        start_time=start_time, endgame=endgame, verbose=True, debug=True
    )

    times_to_store = [1987, 1988, 1989, 1990, 1991, 1992, 2024]
    # Run the Simulation and store data
    data_store = []
    for state in endgame_sim.iter_run(
        end_time=end_time, sampling_interval=sampling_interval
    ):
        if state.current_time in times_to_store:
            if state.current_time == times_to_store[-1]:
                data_store.append(state.get_state_for_age_group(0, 80).sample_seroprevalence((0.80, 0.99)))
            else:
                data_store.append(state.get_state_for_age_group(0, 66).mf_prevalence_in_population())

    return data_store


# this is the function that python will start execution with when run
def run_model_for_fits(iter, abr, kE, psf):
    # num_iters = int(num_iters)
    # max_workers = os.cpu_count() - 1 if num_iters > os.cpu_count() else num_iters
    # max_workers = max(max_workers, 1)

    runSimulations = partial(
        run_simulations,
        # Verbose Output
        verbose=False,
        # How often do we want to output? 1 = every year, 0.5 = every half year, etc.
        sampling_interval=1,
        # The ABR we want to initialize the model with
        abr=int(abr),
        # The kE value we want to initialize the model with
        kE=kE,
        psf=psf,
        # The start time of the model
        start_time=1900,
        # The end time of the model
        end_time=2030,
    )

    # return_data = process_map(
    #     runSimulations, range(num_iters), max_workers=max_workers
    # )

    return_data = runSimulations(i=iter)

    print(return_data)

    return (return_data)