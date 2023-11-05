import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from multiprocessing import cpu_count

from tqdm.contrib.concurrent import process_map

from epioncho_ibm import Params, Simulation
from epioncho_ibm.state.params import BlackflyParams, TreatmentParams
from epioncho_ibm.tools import Data, add_state_to_run_data, write_data_to_csv


def run_sim(i) -> Data:
    params = Params(
        delta_time_days=1,
        year_length_days=366,
        n_people=440,
        blackfly=BlackflyParams(
            delta_h_zero=0.186,
            delta_h_inf=0.003,
            c_h=0.005,
            bite_rate_per_person_per_year=2297,
            gonotrophic_cycle_length=0.0096,
        ),
        sequela_active=[
            "HangingGroin",
            "Atrophy",
            "Blindness",
            "APOD",
            "CPOD",
            "RSD",
            "Depigmentation",
            "SevereItching",
        ],
        treatment=TreatmentParams(
            interval_years=1,
            start_time=100,
            stop_time=140,
        ),
    )

    simulation = Simulation(start_time=0, params=params, verbose=True, debug=True)
    run_data: Data = {}
    for state in simulation.iter_run(
        end_time=140, sampling_years=[i for i in range(100, 140)]
    ):
        add_state_to_run_data(
            state,
            run_data=run_data,
            number=True,
            n_treatments=False,
            achieved_coverage=False,
            with_age_groups=False,
            prevalence=True,
            mean_worm_burden=False,
            intensity=True,
        )
    return run_data


run_iters = 200

if __name__ == "__main__":
    data: list[Data] = process_map(
        run_sim,
        range(run_iters),
        max_workers=cpu_count() + 4,
    )
    write_data_to_csv(data, "70_mfp_no_rebound_dynamics_fixed.csv")


output_data: list[Data] = []
# for j in range(1, n_sims):
#    run_data: Data = {}


# print("Time: " + str(state.current_time))
# print("MF Prev: " + str(state.mf_prevalence_in_population()))
# print("Blidness Prev: " + str(state.sequalae_prevalence()['Blindness']))


#    output_data.append(run_data)
# write_data_to_csv(output_data, '70_mfp_no_rebound_dynamics.csv')
