from typing import Generator
from epioncho_ibm.state import State
import math
import numpy as np

def conduct_survey(state: State):
    if (
        state.stop_survey_workflow_information["total_treatments_given"] >= state._params.min_years_treatment_prestop_survey and
        state.stop_survey_workflow_information["last_sero_prestop_survey"] < np.floor(state.current_time) and # TODO: make yearly surveys parameterized
        state.stop_survey_workflow_information["sero_prestop_reached_time"] < 0
    ):
        prestop_survey(state)
    
    if (
        state.stop_survey_workflow_information["sero_prestop_reached_time"] > 0 and
        (
            state.stop_survey_workflow_information["blackfly_stop_reached_time"] < 0 and
            (
                "retest_blackfly_stop" not in state.stop_survey_workflow_information or
                state.stop_survey_workflow_information["retest_blackfly_stop"] < np.floor(state.current_time)
            )
        )
    ):
        stop_mda_blackfly_survey(state)

    if (
        state.stop_survey_workflow_information["total_treatments_given"] >= state._params.min_years_treatment_stop_survey and
        state.stop_survey_workflow_information["blackfly_stop_reached_time"] > 0 and
        (
            state.stop_survey_workflow_information["sero_stop_survey_reached_time"] < 0 and
            (
                "retest_sero_stop" not in state.stop_survey_workflow_information or
                state.stop_survey_workflow_information["retest_sero_stop"] < np.floor(state.current_time)
            )
        )
    ):
         stop_mda_serological_survey(state)

    if (
        (
            "sero_pts_test" in state.stop_survey_workflow_information and
            "blackfly_pts_test" in state.stop_survey_workflow_information
        ) and
        (
            state.stop_survey_workflow_information["blackfly_pts_test"] <= np.floor(state.current_time) and
            state.stop_survey_workflow_information["sero_pts_test"] <= np.floor(state.current_time)
        ) and 
        (
            state.stop_survey_workflow_information["final_check_pre_who_verification"] < 0
        )
    ):
         pts_survey(state)

def wilson_ci(x, n, alpha=0.05):
    p_hat = x / n
    z = math.sqrt(2) * math.erfcinv(alpha)
    denominator = 1 + z**2 / n
    centre = p_hat + z**2 / (2*n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n)
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return max(0, lower), min(1, upper)

def compare_to_threshold(value: float, threshold: float, specificity: float):
    passed_threshold = False
    # If the specifcity + threshold == 1, then in theory it is almost impossible to go below the threshold
    passed_threshold_max_specificity = threshold + specificity == 1.0 and value <= threshold
    passed_threshold = value < threshold
    return passed_threshold_max_specificity or passed_threshold

def wilson_ci_from_sample_prevalence(bit_generator: Generator, prevalence: float, num_samples: int):
    positive_flies = bit_generator.binomial(
        num_samples, prevalence
    )
    return positive_flies, wilson_ci(positive_flies, num_samples)

def prestop_survey(
        state: State
    ):
        state.stop_survey_workflow_information["last_sero_prestop_survey"] = np.floor(state.current_time)
        prestop_ages = state._params.sero_prestop_survey_age_group
        apparent_seroprev = (
            state
            .get_state_for_age_group(prestop_ages[0], prestop_ages[1])
            .sample_seroprevalence(state._params.survey_serotest_sens_spec)
        )

        if (compare_to_threshold(
            apparent_seroprev, state._params.sero_prestop_survey_threshold, state._params.survey_serotest_sens_spec[1]
        )):
            state.stop_survey_workflow_information["sero_prestop_reached_time"] = state.stop_survey_workflow_information["last_sero_prestop_survey"]

def do_blackfly_survey(state: State):
    l3_prevalence_blackfly = state.calculate_prevalence_l3_blackflies()
    return l3_prevalence_blackfly, wilson_ci_from_sample_prevalence(
        state.derived_params.numpy_bit_generator, l3_prevalence_blackfly,
        state._params.blackfly_stop_sample_size
    )

def stop_mda_blackfly_survey(state: State):
        actual_prev, _, confidence_interval = do_blackfly_survey()

        # Store information for current survey
        curr_retest_num = state.stop_survey_workflow_information["retest_blackfly_stop_count"]
        state.stop_survey_workflow_information[f"blackfly_stop_retest_{curr_retest_num}"] = np.floor(state.current_time)
        state.stop_survey_workflow_information[f"blackfly_stop_retest_actual_prev_{curr_retest_num}"] = actual_prev
        state.stop_survey_workflow_information[f"blackfly_stop_retest_prev_upper_conf_{curr_retest_num}"] = confidence_interval[1]
        
        # TODO: implement an actual sens/spec for the blackfly test
        if compare_to_threshold(confidence_interval[1], state._params.blackfly_stop_threshold, 1):
            state.stop_survey_workflow_information["blackfly_stop_reached_time"] = np.floor(state.current_time)
        else:
            state.stop_survey_workflow_information["retest_blackfly_stop"] = np.floor(state.current_time) + state._params.additional_treatment_years
            state.stop_survey_workflow_information["retest_blackfly_stop_count"] += 1
    
def stop_mda_serological_survey(state: State):
        stop_ages = state._params.sero_stop_survey_age_group
        apparent_seroprev = (
            state
            .get_state_for_age_group(stop_ages[0], stop_ages[1])
            .sample_seroprevalence(state._params.survey_serotest_sens_spec)
        )
        if (compare_to_threshold(
            apparent_seroprev, state._params.sero_stop_survey_threshold, state._params.survey_serotest_sens_spec[1]
        )):
            state.stop_survey_workflow_information["sero_stop_survey_reached_time"] = np.floor(state.current_time)

        # Store information for current test
        curr_retest_num = state.stop_survey_workflow_information["retest_sero_stop_count"]
        state.stop_survey_workflow_information[f"sero_stop_retest_{curr_retest_num}"] = np.floor(state.current_time)
        state.stop_survey_workflow_information[f"sero_stop_retest_prev_{curr_retest_num}"] = apparent_seroprev

        if (state.stop_survey_workflow_information["sero_stop_survey_reached_time"] > 0):
            state.stop_survey_workflow_information["stop_mda_decision_reached"] = 1
            state.stop_survey_workflow_information["blackfly_pts_test"] = np.floor(state.current_time) + state._params.sero_pts_survey_delay
            state.stop_survey_workflow_information["sero_pts_test"] = np.floor(state.current_time) + state._params.sero_pts_survey_delay
        else:
            state.stop_survey_workflow_information["retest_sero_stop"] = np.floor(state.current_time) + state._params.additional_treatment_years
            state.stop_survey_workflow_information["retest_sero_stop_count"] += 1

def pts_survey(state: State):
        state.stop_survey_workflow_information["final_check_pre_who_verification"] = np.floor(state.current_time)
        actual_prev, _, confidence_interval = do_blackfly_survey(state)

        stop_ages = state._params.sero_stop_survey_age_group
        apparent_seroprev = state.get_state_for_age_group(stop_ages[0], stop_ages[1]).sample_seroprevalence(state._params.survey_serotest_sens_spec)
        if (
            compare_to_threshold(apparent_seroprev, state._params.sero_stop_survey_threshold, state._params.survey_serotest_sens_spec[1]) and 
            compare_to_threshold(confidence_interval[1], state._params.blackfly_stop_threshold, 1)
        ):
            state.stop_survey_workflow_information["can_start_who_verification"] = np.floor(state.current_time)

        state.stop_survey_workflow_information["blackfly_pts_actual_prev"] = actual_prev
        state.stop_survey_workflow_information["blackfly_pts_prev_upper_conf_"] = confidence_interval[1]
        state.stop_survey_workflow_information["sero_pts_prev"] = apparent_seroprev