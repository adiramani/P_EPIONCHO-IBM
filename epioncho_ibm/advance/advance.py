import numpy as np

from epioncho_ibm.state import Array, State
from epioncho_ibm.state.sequelae import Sequela

from .blackfly import calc_l1, calc_l2, calc_l3, calc_new_worms_from_blackfly
from .exposure import calculate_total_exposure
from .microfil import calculate_microfil_delta
from .treatment import get_treatment
from .worms import calculate_new_worms


def advance_state(state: State, debug: bool = False) -> None:
    # Pre-Stop Survey
    # TODO: Modularize
    if (
        state.total_treatments_given >= state._params.min_years_treatment_pre_stop_survey and
        state.last_ov16_survey < np.floor(state.current_time) and # TODO: make yearly surveys parameterized
        state.people.stop_mda_workflow_information["sero_pre_stop_reached_time"] < 0
    ):
        state.last_ov16_survey = np.floor(state.current_time)
        pre_stop_ages = state._params.sero_pre_stop_survey_age_group
        apparent_sero_prev = state.get_state_for_age_group(pre_stop_ages[0], pre_stop_ages[1]).sample_seroprevalence(state._params.serotest_sens_spec)
        # If the specifcity + threshold == 1, then in theory it is almost impossible to go below the threshold
        if state._params.sero_pre_stop_survey_threshold + state._params.serotest_sens_spec[1] == 1.0:
            if (apparent_sero_prev <= state._params.sero_pre_stop_survey_threshold):
                state.people.stop_mda_workflow_information["sero_pre_stop_reached_time"] = state.last_ov16_survey
            else:
                if (apparent_sero_prev < state._params.sero_pre_stop_survey_threshold):
                    state.people.stop_mda_workflow_information["sero_pre_stop_reached_time"] = state.last_ov16_survey
        
    if (
        state.people.stop_mda_workflow_information["sero_pre_stop_reached_time"] > 0 and
        (
            state.people.stop_mda_workflow_information["blackfly_stop_reached_time"] < 0 and
            (
                "retest_blackfly_stop" not in state.people.stop_mda_workflow_information or
                state.people.stop_mda_workflow_information["retest_blackfly_stop"] < np.floor(state.current_time)
            )
        )
    ):
        l3_prevalence_blackfly = state.calculate_prevalence_l3_blackflies()
        positive_flies = state.derived_params.numpy_bit_generator.binomial(
            state._params.blackfly_stop_sample_size, l3_prevalence_blackfly
        )
        if positive_flies == 0: # TODO: In theory we would calcualte the upper confidence interval and see if it is less than 0.05%
            state.people.stop_mda_workflow_information["blackfly_stop_reached_time"] = np.floor(state.current_time)
        else:
            state.people.stop_mda_workflow_information["retest_blackfly_stop"] = np.floor(state.current_time) + state._params.additional_treatment_years
    
    stop_mda_decision_reached = False
    if (
        state.total_treatments_given >= state._params.min_years_treatment_stop_survey and
        state.people.stop_mda_workflow_information["blackfly_stop_reached_time"] > 0 and
        (
            state.people.stop_mda_workflow_information["sero_stop_survey_reached_time"] < 0 and
            (
                "retest_sero_stop" not in state.people.stop_mda_workflow_information or
                state.people.stop_mda_workflow_information["retest_sero_stop"] < np.floor(state.current_time)
            )
        )
    ):
        stop_ages = state._params.sero_stop_survey_age_group
        apparent_sero_prev = state.get_state_for_age_group(stop_ages[0], stop_ages[1]).sample_seroprevalence(state._params.serotest_sens_spec)
        # If the specifcity + threshold == 1, then in theory it is almost impossible to go below the threshold
        if state._params.sero_stop_survey_threshold + state._params.serotest_sens_spec[1] == 1.0:
            if (apparent_sero_prev <= state._params.sero_stop_survey_threshold):
                state.people.stop_mda_workflow_information["sero_stop_survey_reached_time"] = np.floor(state.current_time)
            else:
                if (apparent_sero_prev < state._params.sero_stop_survey_threshold):
                   state.people.stop_mda_workflow_information["sero_stop_survey_reached_time"] = np.floor(state.current_time)
        if (state.people.stop_mda_workflow_information["sero_stop_survey_reached_time"] > 0):
            stop_mda_decision_reached = True
            state.people.stop_mda_workflow_information["blackfly_pts_retest"] = np.floor(state.current_time) + state._params.sero_post_stop_survey_delay
            state.people.stop_mda_workflow_information["sero_pts_retest"] = np.floor(state.current_time) + state._params.sero_post_stop_survey_delay
        else:
            state.people.stop_mda_workflow_information["retest_sero_stop"] = np.floor(state.current_time) + state._params.additional_treatment_years
    
    if (
        (
            "sero_pts_retest" in state.people.stop_mda_workflow_information and
            "blackfly_pts_retest" in state.people.stop_mda_workflow_information
        ) and
        (
            state.people.stop_mda_workflow_information["blackfly_pts_retest"] < np.floor(state.current_time) and
            state.people.stop_mda_workflow_information["sero_pts_retest"] < np.floor(state.current_time)
        ) and 
        (
            state.people.stop_mda_workflow_information["final_check_pre_who_verification"] < 0
        )
    ):
        state.people.stop_mda_workflow_information["final_check_pre_who_verification"] = np.floor(state.current_time)
        l3_prevalence_blackfly = state.calculate_prevalence_l3_blackflies()
        positive_flies = state.derived_params.numpy_bit_generator.binomial(
            state._params.blackfly_stop_sample_size, l3_prevalence_blackfly
        )

        stop_ages = state._params.sero_stop_survey_age_group
        apparent_sero_prev = state.get_state_for_age_group(stop_ages[0], stop_ages[1]).sample_seroprevalence(state._params.serotest_sens_spec)
        sero_mda_stop_threshold_reached = False
        # If the specifcity + threshold == 1, then in theory it is almost impossible to go below the threshold
        if state._params.sero_stop_survey_threshold + state._params.serotest_sens_spec[1] == 1.0:
            if (apparent_sero_prev <= state._params.sero_stop_survey_threshold):
                sero_mda_stop_threshold_reached = True
            else:
                if (apparent_sero_prev < state._params.sero_stop_survey_threshold):
                   sero_mda_stop_threshold_reached = True
        if (
            sero_mda_stop_threshold_reached and 
            positive_flies == 0 # TODO: In theory we would calcualte the upper confidence interval and see if it is less than 0.05%
        ):
            state.people.stop_mda_workflow_information["can_start_who_verification"] = np.floor(state.current_time)

    """Advance the state forward one time step from t to t + dt"""
    _, measured_mf = state.microfilariae_per_skin_snip()
    rounded_mf: Array.Person.Float = np.round(measured_mf)
    treatment = get_treatment(
        state._params.treatment,
        state._params.delta_time,
        state.current_time,
        state.derived_params.treatment_times,
        state.derived_params.treatment_index,
        state.people.ages,
        state.people.compliance,
        stop_mda_decision_reached,
        state.derived_params.numpy_bit_generator,
    )
    if treatment is not None and treatment.treatment_occurred:
        state.total_treatments_given += 1
        state.derived_params.treatment_index += 1
        assert state.n_treatments is not None
        n_people_by_age, _ = np.histogram(
            state.people.ages,
            bins=np.arange(0, state._params.humans.max_human_age + 1),
        )
        n_treatments_by_age, _ = np.histogram(
            state.people.ages[treatment.coverage_in],
            bins=np.arange(0, state._params.humans.max_human_age + 1),
        )

        treatment_name_val = str(state._params.treatment.treatment_name) + " MDA Round"
        state.n_treatments[
            str(state.current_time) + "," + str(treatment_name_val)
        ] = n_treatments_by_age
        state.n_treatments_population[
            str(state.current_time) + "," + str(treatment_name_val)
        ] = n_people_by_age

        state.people.has_been_treated = (
            state.people.has_been_treated | treatment.coverage_in
        )

    total_exposure = calculate_total_exposure(
        state._params.exposure,
        state._params.humans.gender_ratio,
        state.people.ages,
        state.people.sex_is_male,
        state.people.individual_exposure,
    )
    old_ages = state.people.ages.copy()
    state.people.ages += state._params.delta_time

    old_worms = state.people.worms.copy()

    # there is a delay in new parasites entering humans (from fly bites) and
    # entering the first adult worm age class
    new_worms = calc_new_worms_from_blackfly(
        state.people.blackfly.L3,
        state._params.blackfly,
        state._params.delta_time,
        total_exposure,
        state.n_people,
        debug,
        state.derived_params.numpy_bit_generator,
    )

    if state.people.delay_arrays.worm_delay is None:
        worm_delay: Array.Person.Int = new_worms
    else:
        worm_delay: Array.Person.Int = state.people.delay_arrays.worm_delay

    state.people.worms, state.people.last_treatment = calculate_new_worms(
        current_worms=state.people.worms,
        worm_params=state._params.worms,
        treatment=treatment,
        last_treatment=state.people.last_treatment,
        delta_time=state._params.delta_time,
        worm_delay_array=worm_delay,
        mortalities=state.derived_params.worm_mortality_rate,
        mortalities_generator=state.derived_params.worm_mortality_generator,
        current_time=state.current_time,
        debug=debug,
        mortality_rate=state.derived_params.worm_mortality_rate,
        worm_age_rate_generator=state.derived_params.worm_age_rate_generator,
        worm_sex_ratio_generator=state.derived_params.worm_sex_ratio_generator,
        worm_lambda_zero_generator=state.derived_params.worm_lambda_zero_generator,
        worm_omega_generator=state.derived_params.worm_omega_generator,
        numpy_bit_gen=state.derived_params.numpy_bit_generator,
    )

    # inputs for delay in L1
    total_mf: Array.Person.Float = np.sum(state.people.mf, axis=0)
    state.people.mf += calculate_microfil_delta(
        current_microfil=state.people.mf,
        delta_time=state._params.delta_time,
        microfil_params=state._params.microfil,
        treatment_params=state._params.treatment,
        microfillarie_mortality_rate=state.derived_params.microfillarie_mortality_rate,
        fecundity_rates_worms=state.derived_params.fecundity_rates_worms,
        last_treatment=state.people.last_treatment,
        current_time=state.current_time,
        current_fertile_female_worms=old_worms.fertile,
        current_male_worms=old_worms.male,
        debug=debug,
    )

    old_blackfly_L1 = state.people.blackfly.L1

    if state.people.delay_arrays.exposure_delay is None:
        exposure_delay: Array.Person.Float = total_exposure
    else:
        exposure_delay: Array.Person.Float = state.people.delay_arrays.exposure_delay

    if state.people.delay_arrays.mf_delay is None:
        mf_delay: Array.Person.Float = total_mf.copy()
    else:
        mf_delay: Array.Person.Float = state.people.delay_arrays.mf_delay

    state.people.was_infected |= state.people.get_infected()

    state._update_for_epilepsy()

    state.people.blackfly.L1 = calc_l1(
        state._params.blackfly,
        total_mf,
        mf_delay,
        total_exposure,
        exposure_delay,
        state._params.year_length_days,
    )

    old_blackfly_L2 = state.people.blackfly.L2
    state.people.blackfly.L2 = calc_l2(
        state._params.blackfly,
        old_blackfly_L1,
        mf_delay,
        exposure_delay,
        state._params.year_length_days,
    )
    state.people.blackfly.L3 = calc_l3(state._params.blackfly, old_blackfly_L2)
    # TODO: Resolve new_mf=old_mf
    state.people.delay_arrays.lag_all_arrays(
        new_worms=new_worms, total_exposure=total_exposure, new_mf=total_mf
    )

    new_has_sequela = {}
    _, new_measured_mf = state.microfilariae_per_skin_snip()
    new_rounded_mf: Array.Person.Float = np.round(new_measured_mf)
    new_total_mf: Array.Person.Float = np.sum(state.people.mf, axis=0)
    for name, old_rel_sequela in state.people.has_sequela.items():
        seq_class = state.derived_params.sequela_classes[name]
        assert name in state.people.countdown_sequela
        rel_seq_countdown = state.people.countdown_sequela[name]
        prob = seq_class.timestep_probability(
            delta_time=state._params.delta_time,
            true_mf_count=new_total_mf,
            measured_mf_count=new_rounded_mf,
            ages=old_ages,
            existing_sequela=state.people.has_sequela,
            has_this_sequela=old_rel_sequela,
            countdown=rel_seq_countdown,
        )

        new_condition = np.random.random(state.n_people) < prob

        if seq_class.end_countdown_become_positive is not None:
            assert seq_class.years_countdown is not None

            # Decrement countdown
            rel_seq_countdown[rel_seq_countdown > 0] -= state._params.delta_time

            # For now infected, start countdown
            rel_seq_countdown[new_condition] = seq_class.years_countdown

            countdown_up = np.round(rel_seq_countdown, 4) <= 0
            # Reset countdown with countdown up
            rel_seq_countdown[countdown_up] = np.inf

            # Set those with countdown up to correct state

            if seq_class.end_countdown_become_positive:
                # Lagged case
                new_has_sequela[name] = old_rel_sequela
                new_has_sequela[name][countdown_up] = True
            else:
                # Reversible Case
                new_has_sequela[name] = old_rel_sequela | new_condition
                new_has_sequela[name][countdown_up] = False
        else:
            new_has_sequela[name] = old_rel_sequela | new_condition

    state.people.has_sequela = new_has_sequela
    new_seropositives = np.logical_and(
        (np.sum(state.people.mf, axis=0) > 0),
        np.logical_and(
            (np.sum(state.people.worms.fertile, axis=0) > 0),
            (np.sum(state.people.worms.male, axis=0) > 0)
        )
    )
    state.people.ov16_serostatus[new_seropositives] = True

    people_to_die: Array.Person.Bool = np.logical_or(
        state.derived_params.people_to_die_generator.binomial(
            n=np.repeat(1, state.n_people), p=state._params.delta_time / state._params.humans.mean_human_age
        )
        == 1,
        state.people.ages >= state._params.humans.max_human_age,
    )
    state.people.process_deaths(
        people_to_die,
        state._params.humans.gender_ratio,
        state.derived_params.numpy_bit_generator,
        state._params.treatment,
        state._params.gamma_distribution,
    )
