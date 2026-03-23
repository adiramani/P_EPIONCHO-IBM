import math
import os
import sys
from functools import partial

import h5py
from tqdm.contrib.concurrent import process_map

from epioncho_ibm.endgame_simulation import EndgameSimulation
from epioncho_ibm.state.params import EpionchoEndgameModel
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


# You can edit the inputs to this function to set more parameters dynamically
def get_parameters(iter, abr=1641, kE=0.3, psf=0.5, x0=0, x1=1, hbi_lb=0, current_day_intervention="aIVM"):
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
            "last_year": 2026,
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

    curr_hbi = max(math.exp(x0 + x1 * (abr * 0.2)), hbi_lb)
    changes.append(
        {
            "year": 1986,
            "params": {"blackfly": {
                "bite_rate_per_person_per_year": abr * 0.2,
                "delta_h_zero": delta_hz_in,
                "c_h": c_h_in,
                "delta_h_inf": delta_hinf_in,
                "x0": x0,
                "x1": x1,
                "derive_human_blood_index": False,
                "human_blood_index": curr_hbi,
                "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
            }},
        }
    )
    curr_hbi = max(math.exp(x0 + x1 * abr), hbi_lb)
    changes.append(
        {
            "year": 1987,
            "params": {"blackfly": {
                "bite_rate_per_person_per_year": abr,
                "delta_h_zero": delta_hz_in,
                "c_h": c_h_in,
                "delta_h_inf": delta_hinf_in,
                "x0": x0,
                "x1": x1,
                "derive_human_blood_index": False,
                "human_blood_index": curr_hbi,
                "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
            }},
        }
    )
    curr_hbi = max(math.exp(x0 + x1 * (abr * 0.2)), hbi_lb)
    changes.append(
        {
            "year": 1990,
            "params": {"blackfly": {
                "bite_rate_per_person_per_year": abr * 0.2,
                "delta_h_zero": delta_hz_in,
                "c_h": c_h_in,
                "delta_h_inf": delta_hinf_in,
                "x0": x0,
                "x1": x1,
                "derive_human_blood_index": False,
                "human_blood_index": curr_hbi,
                "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
            }},
        }
    )
    curr_hbi = max(math.exp(x0 + x1 * (abr * 0.36)), hbi_lb)
    changes.append(
        {
            "year": 1996,
            "params": {"blackfly": {
                "bite_rate_per_person_per_year": abr * 0.36,
                "delta_h_zero": delta_hz_in,
                "c_h": c_h_in,
                "delta_h_inf": delta_hinf_in,
                "x0": x0,
                "x1": x1,
                "derive_human_blood_index": False,
                "human_blood_index": curr_hbi,
                "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
            }},
        }
    )
    curr_hbi = max(math.exp(x0 + x1 * (abr * 0.52)), hbi_lb)
    changes.append(
        {
            "year": 1997,
            "params": {"blackfly": {
                "bite_rate_per_person_per_year": abr * 0.52,
                "delta_h_zero": delta_hz_in,
                "c_h": c_h_in,
                "delta_h_inf": delta_hinf_in,
                "x0": x0,
                "x1": x1,
                "derive_human_blood_index": False,
                "human_blood_index": curr_hbi,
                "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
            }},
        }
    )
    curr_hbi = max(math.exp(x0 + x1 * (abr * 0.68)), hbi_lb)
    changes.append(
        {
            "year": 1998,
            "params": {"blackfly": {
                "bite_rate_per_person_per_year": abr * 0.68,
                "delta_h_zero": delta_hz_in,
                "c_h": c_h_in,
                "delta_h_inf": delta_hinf_in,
                "x0": x0,
                "x1": x1,
                "derive_human_blood_index": False,
                "human_blood_index": curr_hbi,
                "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
            }},
        }
    )
    curr_hbi = max(math.exp(x0 + x1 * (abr * 0.84)), hbi_lb)
    changes.append(
        {
            "year": 1999,
            "params": {"blackfly": {
                "bite_rate_per_person_per_year": abr * 0.84,
                "delta_h_zero": delta_hz_in,
                "c_h": c_h_in,
                "delta_h_inf": delta_hinf_in,
                "x0": x0,
                "x1": x1,
                "derive_human_blood_index": False,
                "human_blood_index": curr_hbi,
                "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
            }},
        }
    )
    curr_hbi = max(math.exp(x0 + x1 * abr), hbi_lb)
    changes.append(
        {
            "year": 2000,
            "params": {"blackfly": {
                "bite_rate_per_person_per_year": abr,
                "delta_h_zero": delta_hz_in,
                "c_h": c_h_in,
                "delta_h_inf": delta_hinf_in,
                "x0": x0,
                "x1": x1,
                "derive_human_blood_index": False,
                "human_blood_index": curr_hbi,
                "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
            }},
        }
    )

    if (current_day_intervention != ""):
        frequency = 0.5 if "b" in current_day_intervention else 1
        min_age_treat = 5
        mf_nu = 0.0096
        mf_omega = 1.25
        emb_lamb_max = 32.4
        emb_phi = 19.6
        perma_inf = 0.345
        treat_name = "IVM"
        if "MOX" in current_day_intervention:
            treat_name = "MOX"
            min_age_treat = 4
            mf_nu = 0.04
            mf_omega = 1.82
            emb_lamb_max = 462
            emb_phi = 4.83

            changes.append(
                {
                    "year": 2027, 
                    "params": {
                        "delta_time_days": 0.5,
                        "run_stop_mda_workflow": True,
                        "additional_treatment_years": 1,
                        "sero_stop_survey_age_group": [5, 10],
                        "sero_stop_survey_threshold": 0.015,
                        "blackfly": {
                            "bite_rate_per_person_per_year": abr,
                            "delta_h_zero": delta_hz_in,
                            "c_h": c_h_in,
                            "delta_h_inf": delta_hinf_in,
                            "x0": x0,
                            "x1": x1,
                            "derive_human_blood_index": False,
                            "human_blood_index": curr_hbi,
                            "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
                        }    
                    }
                }
            )

        treatment_program.append(
            {
                "first_year": 2027,
                "last_year": 2050,
                "treatment_name": treat_name,
                "interventions": {
                    "treatment_interval": frequency,
                    "total_population_coverage": 0.79,
                    "correlation": 0.2,
                    "min_age_of_treatment": min_age_treat,
                    "microfilaricidal_nu": mf_nu,
                    "microfilaricidal_omega": mf_omega,
                    "embryostatic_lambda_max": emb_lamb_max,
                    "embryostatic_phi": emb_phi,
                    "permanent_infertility": perma_inf,
                },
            }
        )

    return {
        "parameters": {
            "initial": {
                "n_people": 400,
                "year_length_days": 365,
                "seed": seed,
                "gamma_distribution": kE,
                "delta_time_days": 1,
                "run_stop_mda_workflow": True,
                "additional_treatment_years": 1,
                "sero_stop_survey_age_group": [5, 10],
                "sero_stop_survey_threshold": 0.015,
                "blackfly": {
                    "delta_h_zero": delta_hz_in,
                    "c_h": c_h_in,
                    "delta_h_inf": delta_hinf_in,
                    "bite_rate_per_person_per_year": abr,
                    "x0": x0,
                    "x1": x1,
                    "derive_human_blood_index": False,
                    "human_blood_index": curr_hbi,
                    "bite_rate_per_fly_on_human": curr_hbi / (1 / 104)
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
    x0 = 0,
    x1 = 1,
    hbi_lb = 0,
    int_type = "aIVM",
    start_time=1900,
    end_time=2005,
    capture_all_data=False
):
    endgame_structure = get_parameters(i, abr=abr, kE=kE, psf=psf, x0=x0, x1=x1, hbi_lb=hbi_lb, current_day_intervention=int_type)

    # Read in endgame objects and set up simulation
    endgame = EpionchoEndgameModel.parse_obj(endgame_structure)

    # EndgameSimulation is a type of Simulation that allows for changes to the parameters
    endgame_sim = EndgameSimulation(
        start_time=start_time, endgame=endgame, verbose=verbose, debug=True
    )

    times_to_store = [1987, 1988, 1989, 1990, 1991, 1992, 2024]
    # Run the Simulation and store data
    data_store = []
    run_data = {}
    for state in endgame_sim.iter_run(
        end_time=end_time, sampling_interval=sampling_interval
    ):
        if not capture_all_data:
            if state.current_time in times_to_store:
                if state.current_time == times_to_store[-1]:
                    data_store.append(state.get_state_for_age_group(0, 80).sample_seroprevalence((0.80, 0.99)))
                else:
                    data_store.append(state.get_state_for_age_group(0, 66).mf_prevalence_in_population())
        else:
            add_state_to_run_data(
                state,
                run_data,
                prevalence=True,
                intensity=True,
                number=True,
                with_age_groups=True,
                with_ov16=True,
                with_blackfly_outputs=True,
                mean_worm_burden = True,
                with_stop_mda_information = True,
                custom_age_groups=[
                    (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40),
                    (40, 50), (50, 60), (60, 70), (70, 81),  (0, 66), (0, 80)
                ],
                ov16_sens_spec=(0.80, 0.99)
            )

    if capture_all_data:
        return run_data
    return data_store


# this is the function that python will start execution with when run
def run_model_for_fits(
        cores, num_iters, abr, kE, psf, x0, x1,
        hbi_lower_bound, treat_type, verbose=False, capture_all_data=False,
    ):
    num_iters = int(num_iters)
    cores = int(cores)

    runSimulations = partial(
        run_simulations,
        # Verbose Output
        verbose=verbose,
        # How often do we want to output? 1 = every year, 0.5 = every half year, etc.
        sampling_interval=1,
        # The ABR we want to initialize the model with
        abr=int(abr),
        # The kE value we want to initialize the model with
        kE=kE,
        psf=psf,
        x0=x0,
        x1=x1, 
        hbi_lb = hbi_lower_bound,
        int_type = treat_type,
        # The start time of the model
        start_time=1900,
        # The end time of the model
        end_time=2100,
        capture_all_data=capture_all_data
    )

    if num_iters == 1:
        return_data = runSimulations(i=1)
    return_data = process_map(
        runSimulations, range(num_iters), max_workers=cores
    )

    return (return_data)

def run_model_single_param():
    abrs = [43526, 38690, 15969, 14965, 11589, 8395, 8304, 8121, 4745, 4380, 3011, 2920, 1551, 730, 548, 365, 91]
    for abr in abrs:
        kE = 0.3
        if abr > 8000:
            kE = 0.4
        if abr < 1000:
            kE = 0.2
        model_output = run_model_for_fits(
            10, 10, abr, kE, 0.79, -0.39, -0.0000447, 0.15, 
            "bMOX", capture_all_data=True
        )
        write_data_to_csv(
            model_output,
            f"test_outputs/python_model_output/test_elimination/testing_best_fit_params_{abr}.csv",
        )

if __name__ == "__main__":
    run_model_single_param()