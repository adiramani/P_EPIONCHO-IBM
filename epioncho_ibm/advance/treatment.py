from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

from epioncho_ibm.state import Array, HumanParams, State, TreatmentParams


def _is_during_treatment(
    treatment: TreatmentParams,
    current_time: float,
    delta_time: float,
    treatment_times: Optional[Array.Treatments.Float],
) -> tuple[bool, bool]:
    """
    Returns two booleans describing if treatment has started, and if it occurred.

    Args:
        treatment_params (TreatmentParams | None): The fixed parameters relating to treatment
        current_time (float): The current time, t, in the model
        delta_time (float): dt - The amount of time advance in one time step
        treatment_times (Optional[Array.Treatments.Float]): The times for treatment across the model.

    Returns:
        tuple[bool, bool]: bool describing if treatment started,
            bool describing if treatment occurred, respectively
    """
    treatment_started = current_time >= treatment.start_time
    if treatment_started:
        assert treatment_times is not None
        treatment_occurred: bool = (
            bool(
                np.any(
                    (treatment_times <= current_time)
                    & (treatment_times > current_time - delta_time)
                )
            )
            and current_time <= treatment.stop_time
        )
        if (current_time > 2024) and (current_time % 1 < (1 / 366)):
            if current_time == 2024 + delta_time:
                print(
                    "Treatment Checks: Treatment should occur on the first timestep of each year\n"
                )
            if current_time < 2026:
                print("IVM Treatment with old delta time")
            else:
                print("MOX Treatment with new delta time")
            print("Current Time: ", str(current_time))
            print("Delta Time: ", str(delta_time))
            print("Will treatment happen: ", str(treatment_occurred) + "\n")
    else:
        treatment_occurred = False
    return treatment_started, treatment_occurred


@dataclass
class TreatmentGroup:
    """
    treatment_params (TreatmentParams): The fixed parameters relating to treatment
    coverage_in (Array.Person.Bool): An array stating if each person in the model is treated
    treatment_times (Array.Treatments.Float): The times for treatment across the model.
    treatment_occurred (bool): A boolean stating if treatment occurred in this time step.
    """

    treatment_params: TreatmentParams
    coverage_in: Array.Person.Bool
    treatment_times: Array.Treatments.Float
    treatment_occurred: bool


def get_treatment(
    treatment_params: Optional[TreatmentParams],
    delta_time: float,
    current_time: float,
    treatment_times: Optional[Array.Treatments.Float],
    ages: Array.Person.Float,
    compliance: Optional[Array.Person.Float],
    numpy_bit_gen: Generator,
    state: State,
) -> Optional[TreatmentGroup]:
    """
    Generates a treatment group, and calculates coverage, based on the current time

    Args:
        treatment_params (TreatmentParams | None): The fixed parameters relating to treatment
        delta_time (float): dt - The amount of time advance in one time step
        current_time (float): The current time, t, in the model
        treatment_times (Optional[Array.Treatments.Float]): The times for treatment across the model.
        ages (Array.Person.Float): The ages of the people
        compliance (Array.Person.Float): The compliance of the people
        numpy_bit_gen: (Generator): The random number generator for numpy

    Returns:
        Optional[TreatmentGroup]: A treatment group containing information for later treatment
            calculation
    """
    if treatment_params is not None:
        assert compliance is not None
        assert treatment_times is not None
        treatment_started, treatment_occurred = _is_during_treatment(
            treatment_params,
            current_time,
            delta_time,
            treatment_times,
        )
        if treatment_started:
            state.people.update_zero_compliance(
                treatment_params.correlation,
                treatment_params.total_population_coverage,
                state.numpy_bit_generator,
            )
            compliance = state.people.compliance
            rand_nums = numpy_bit_gen.uniform(low=0, high=1, size=len(ages))
            too_young = ages < treatment_params.min_age_of_treatment
            return TreatmentGroup(
                treatment_params=treatment_params,
                coverage_in=(rand_nums < compliance) & ~too_young,
                treatment_times=treatment_times,
                treatment_occurred=treatment_occurred,
            )
        else:
            return None
    else:
        return None
