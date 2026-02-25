from dataclasses import field
from pathlib import Path
from typing import IO, Callable, Optional, overload

import numpy as np
from endgame_simulations.simulations import BaseState
from hdf5_dataclass import HDF5Dataclass
from numpy.random import Generator
from pydantic import BaseModel
from scipy.optimize import curve_fit

from .derived_params import DerivedParams
from .params import (
    ImmutableParams,
    Params,
    SpecificTreatmentParams,
    immutable_to_mutable,
    mutable_to_immutable,
)
from .people import People
from .prob_mapping import mf_probs
from .types import Array

np.seterr(all="ignore")


class NumericArrayStat(BaseModel):
    mean: float
    # st_dev: float

    @classmethod
    def from_array(cls, array: Array.Person.Float | Array.Person.Int):
        return cls(mean=float(np.mean(array)))  # , st_dev=np.std(array))


class StateStats(BaseModel):
    percent_compliant: float
    percent_male: float
    L1: float
    L2: float
    L3: float
    ages: float
    mf: float
    male_worms: float
    infertile_female_worms: float
    fertile_female_worms: float
    mf_per_skin_snip: float
    population_prevalence: float


def negative_binomial_alt_interface(
    n: Array.General.Float, mu: Array.General.Float, numpy_bit_gen: Generator
) -> Array.General.Int:
    """
    Provides an alternate interface for random negative binomial.

    Args:
        n (Array.General.Float): Number of successes
        mu (Array.General.Float): Mean of the distribution
        numpy_bit_gen: (Generator): The random number generator for numpy

    Returns:
        Array.General.Int: Samples from a negative binomial distribution
    """
    non_zero_n = n[n > 0]
    rel_prob = non_zero_n / (non_zero_n + mu[n > 0])
    temp_output = numpy_bit_gen.negative_binomial(
        n=non_zero_n, p=rel_prob, size=len(non_zero_n)
    )
    output = np.zeros(len(n), dtype=int)
    output[n > 0] = temp_output
    return output


def recalculate_compliance(
    is_compliant: Array.Person.Bool,
    prev_noncompliance_rate: float,
    new_noncompliance_rate: float,
    people_generator: Generator,
):
    if prev_noncompliance_rate == new_noncompliance_rate:
        return None
    elif prev_noncompliance_rate > new_noncompliance_rate:
        new_draw_rate = new_noncompliance_rate / prev_noncompliance_rate
        non_comps = len(is_compliant) - sum(is_compliant)
        new_compliant = people_generator.uniform(low=0, high=1, size=non_comps) < (
            1 - new_draw_rate
        )
        is_compliant[~is_compliant] = new_compliant
        return None
    else:
        new_draw_from_pop = (new_noncompliance_rate - prev_noncompliance_rate) / (
            1 - prev_noncompliance_rate
        )
        comps = sum(is_compliant)
        new_compliant = people_generator.uniform(low=0, high=1, size=comps) < (
            1 - new_draw_from_pop
        )
        is_compliant[is_compliant] = new_compliant
        return None


@overload
def _mf_fit_func(x: float, a: float, b: float) -> float:
    ...


@overload
def _mf_fit_func(
    x: Array.Person.Int | Array.Person.Float, a: float, b: float
) -> Array.Person.Float:
    ...


def _mf_fit_func(
    x: float | Array.Person.Int | Array.Person.Float, a: float, b: float
) -> float | Array.Person.Float:
    return a * (1 + x) ** b


def get_OAE_mf_count_func2(mf: list[int], prob: list[float], val_for_0: float):
    # Use scipy's curve_fit to fit the function to the data
    p_optimal, _ = curve_fit(_mf_fit_func, mf, prob, p0=[0.02, 0.4])
    a = p_optimal[0]
    b = p_optimal[1]

    def new_mf_fit(mf: Array.Person.Int | Array.Person.Float) -> Array.Person.Float:
        mf_probs = np.zeros_like(mf, dtype=float)
        mf_zero_idxs = np.equal(mf, 0)
        mf_probs[mf_zero_idxs] = np.ones_like(mf[mf_zero_idxs], dtype=float) * val_for_0
        mf_probs[~mf_zero_idxs] = _mf_fit_func(x=mf[~mf_zero_idxs], a=a, b=b)
        return mf_probs

    return new_mf_fit


def get_OAE_mf_count_func(mf: list[int], prob: list[float], val_for_0: float):
    input_arr = np.array(mf_probs, dtype=float)

    def new_mf_fit(mf: Array.Person.Int | Array.Person.Float) -> Array.Person.Float:
        int_arr = np.array(np.round(mf), dtype=int)
        return input_arr[int_arr]

    return new_mf_fit


class State(HDF5Dataclass, BaseState[Params]):
    people: People
    _params: ImmutableParams
    n_treatments: Optional[dict[float, Array.General.Int]]
    n_treatments_population: Optional[dict[float, Array.General.Float]]
    stop_survey_workflow_information: dict[str, int]
    current_time: float = 0.0
    _previous_delta_time: Optional[float] = None
    derived_params: DerivedParams = field(init=False, repr=False)
    fit_func_OAE: Callable[
        [Array.Person.Int | Array.Person.Float], Array.Person.Float
    ] = field(init=False, repr=False)

    def __post_init__(self):
        self._derive_params({})
        MF_COUNTS = [3, 13, 36, 76, 151, 200]

        PROB = [0.0439, 0.072, 0.0849, 0.1341, 0.1538, 0.2]

        MF_0_VAL = 0.0061

        self.fit_func_OAE = get_OAE_mf_count_func(MF_COUNTS, PROB, MF_0_VAL)

    @property
    def n_people(self):
        """
        The number of people simulated.
        """
        return len(self.people)

    def get_params(self) -> Params:
        return immutable_to_mutable(self._params)

    def reset_params(self, params: Params):
        """Reset the parameters

        Args:
            params (Params): New set of parameters
        """
        all_generators = self._collect_generators()
        generators_to_keep = self._determine_generators_to_keep(all_generators, self._params.seed, params.seed)

        # backwards compatibility check, where n_treatments used to be an array, instead of a dict
        if not isinstance(self.n_treatments, dict):
            self.n_treatments = {}
            self.n_treatments_population = {}

        # backwards compatibility check, where people.has_been_treated did not exist
        if self.people.has_been_treated is None:
            self.people.has_been_treated = np.full(params.n_people, False)

        old_params = self.get_params()
        self._params = mutable_to_immutable(params)
        self._derive_params(generators_to_keep)
        self.update_compliance(old_params, params)
        self.people.delay_arrays.rescale_delay_arrays(params)

    def _derive_params(self, oldGenerators) -> None:
        assert self._params
        self.set_survey_information_dict()
        self.derived_params = DerivedParams(
            immutable_to_mutable(self._params), self.current_time, oldGenerators
        )

    def set_survey_information_dict(self) -> None:
        if (
            self.stop_survey_workflow_information is None or
            self.stop_survey_workflow_information == {}
        ):
            self.stop_survey_workflow_information = {
                "sero_prestop_reached_time": -1,
                "blackfly_stop_reached_time": -1,
                "sero_stop_survey_reached_time": -1,
                "can_start_who_verification": -1,
                "total_treatments_given": 0,
                "last_sero_prestop_survey": 0,
                "stop_mda_decision_reached": 0,
                "retest_sero_stop_count": 0,
                "retest_blackfly_stop_count": 0,
                "final_check_pre_who_verification": -1
            }

    def _collect_generators(self) -> dict[str, Generator]:
        generators = {}
        for generator_name in self.derived_params.GENERATOR_NAMES:
            generators[generator_name] = getattr(self.derived_params, generator_name, None)
        return generators
    
    def _determine_generators_to_keep(self, generators_dict, old_seed, new_seed) -> dict[str, Generator]:
        for generator_name in generators_dict.keys():
            if (generators_dict[generator_name] is not None) and (old_seed != new_seed):
                generators_dict[generator_name] = None
        return generators_dict
    
    def update_compliance(self, old_params: Params, new_params: Params):
        if new_params.treatment is None:
            self.people.compliance = np.zeros(new_params.n_people)
        elif (
            (old_params.treatment is None)
            or (old_params.treatment.correlation != new_params.treatment.correlation)
            or (
                old_params.treatment.total_population_coverage
                != new_params.treatment.total_population_coverage
            )
        ):
            self.people.update_treatment_prob(
                new_params.treatment.correlation,
                new_params.treatment.total_population_coverage,
                self.derived_params.numpy_bit_generator,
            )

    def get_state_for_age_group(self, age_start: float, age_end: float) -> "State":
        return State(
            people=self.people.get_people_for_age_group(age_start, age_end),
            _params=self._params,
            current_time=self.current_time,
            _previous_delta_time=self._previous_delta_time,
            n_treatments={
                key: value[age_start:age_end]
                for key, value in self.n_treatments.items()
            },
            n_treatments_population={
                key: value[age_start:age_end]
                for key, value in self.n_treatments_population.items()
            },
            stop_survey_workflow_information=self.stop_survey_workflow_information
        )

    @classmethod
    def from_params(
        cls,
        params: Params,
        current_time: float = 0.0,
    ) -> "State":
        """Generate the initial state of the model, based on parameters.

        Args:
            params (Params): A set of fixed parameters for controlling the model.
            current_time (float): Current time of the simulation's state. Defaults to 0

        Returns:
            State: The state of the model
        """
        return cls(
            people=People.from_params(params),
            _params=mutable_to_immutable(params),
            current_time=current_time,
            _previous_delta_time=None,
            n_treatments={},
            n_treatments_population={},
            stop_survey_workflow_information = {},
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, State)
            and self.people == other.people
            and self._params == other._params
            and self.current_time == other.current_time
            and self._previous_delta_time == other._previous_delta_time
        )

    def reset_treatment_counter(self):
        self.n_treatments = {}
        self.n_treatments_population = {}

    def get_treatment_count_for_age_group(
        self, age_start: float, age_end: float
    ) -> int:
        if age_start > age_end:
            raise ValueError(f"Age start {age_start} > age end {age_end}")

        if self.n_treatments is None:
            raise ValueError("Cannot get treatment count for state age sub group")
        return {
            key: np.nansum(value[age_start:age_end])
            for key, value in self.n_treatments.items()
        }

    def get_achieved_coverage_for_age_group(
        self, age_start: float, age_end: float
    ) -> int:
        if age_start > age_end:
            raise ValueError(f"Age start {age_start} > age end {age_end}")

        return {
            key: (
                np.nansum(value[age_start:age_end])
                / np.nansum(self.n_treatments_population[key][age_start:age_end])
            )
            for key, value in self.n_treatments.items()
        }

    def stats(self) -> StateStats:
        if self.people.compliance is not None:
            mean_comp = float(np.sum(self.people.compliance)) / len(
                self.people.compliance
            )
        else:
            mean_comp = 0
        return StateStats(
            percent_compliant=mean_comp,
            percent_male=float(np.sum(self.people.sex_is_male))
            / len(self.people.sex_is_male),
            L1=np.mean(self.people.blackfly.L1),
            L2=np.mean(self.people.blackfly.L2),
            L3=np.mean(self.people.blackfly.L3),
            ages=np.mean(self.people.ages),
            mf=np.mean(self.people.mf),
            male_worms=np.mean(self.people.worms.male),
            infertile_female_worms=np.mean(self.people.worms.infertile),
            fertile_female_worms=np.mean(self.people.worms.fertile),
            mf_per_skin_snip=self.microfilariae_per_skin_snip()[0],
            population_prevalence=self.mf_prevalence_in_population(),
        )

    def microfilariae_per_skin_snip(
        self, return_nan: bool = False
    ) -> tuple[float, Array.Person.Float]:
        """
        Calculates number of mf in skin snip for all people.

        People are tested for the presence of mf using a skin snip.
        We assume mf are overdispersed in the skin

        Returns:
            tuple[float, Array.Person.Float]: Mean mf, mf by person.
        """
        kmf = (
            self._params.microfil.slope_kmf
            * np.sum(
                self.people.worms.fertile + self.people.worms.infertile,
                axis=0,
            )
            + self._params.microfil.initial_kmf
        )

        mu = self._params.humans.skin_snip_weight * np.sum(self.people.mf, axis=0)
        if self._params.humans.skin_snip_number > 1:
            total_skin_snip_mf = np.zeros(
                (
                    self.n_people,
                    self._params.humans.skin_snip_number,
                )
            )
            for i in range(self._params.humans.skin_snip_number):
                total_skin_snip_mf[:, i] = negative_binomial_alt_interface(
                    n=kmf, mu=mu, numpy_bit_gen=self.derived_params.numpy_bit_generator
                )
            mfobs: Array.Person.Int = np.sum(total_skin_snip_mf, axis=1)
        else:
            mfobs: Array.Person.Int = negative_binomial_alt_interface(
                n=kmf, mu=mu, numpy_bit_gen=self.derived_params.numpy_bit_generator
            )

        mfobs_percent: Array.Person.Float = mfobs / (
            self._params.humans.skin_snip_number * self._params.humans.skin_snip_weight
        )
        if mfobs_percent.size == 0:
            if return_nan:
                return np.nan, mfobs_percent
            else:
                return 0.0, mfobs_percent
        else:
            return float(np.mean(mfobs_percent)), mfobs_percent

    def mf_prevalence_in_population(self, return_nan: bool = False) -> float:
        """
        Calculates mf prevalence in population.

        Returns:
            float: mf_prevalence
        """
        pop_over_min_age_array = (
            self.people.ages >= self._params.humans.min_skinsnip_age
        )
        _, mf_skin_snip = self.microfilariae_per_skin_snip()
        infected_over_min_age = int(np.sum(mf_skin_snip[pop_over_min_age_array] > 0))
        total_over_min_age = int(np.sum(pop_over_min_age_array))
        try:
            return infected_over_min_age / total_over_min_age
        except ZeroDivisionError:
            if return_nan:
                return np.nan
            else:
                return 0.0

    def worm_burden_per_person(self, worm_type="all") -> Array.Person.Int:
        """
        Calculated the worm burden of a given type(s) for each host in the population
        
        @param worm_type: can be one of "all", "female", "male", or fertile_female". Denotes which type of worm
            will be used in the denominator of the proportion. "all" will use all worm types.
        Returns:
            list of length # of hosts containing the number of specified worm type in each host.
        """
        if worm_type == "male":
            return self.people.worms.male.sum(0)
        elif worm_type == "female":
            return (
                self.people.worms.fertile.sum(0)
                + self.people.worms.infertile.sum(0)
            )
        elif worm_type == "fertile_female":
            return self.people.worms.fertile.sum(0)
        return (
            self.people.worms.male.sum(0)
            + self.people.worms.fertile.sum(0)
            + self.people.worms.infertile.sum(0)
        )

    def mean_worm_burden(self, worm_type="all") -> float:
        worm_burden = self.worm_burden_per_person(worm_type)
        if worm_burden.size == 0:
            return 0.0
        else:
            return float(np.mean(worm_burden))

    def worm_prevalence(self, worm_type="all") -> float:
        worm_burden = self.worm_burden_per_person(worm_type)
        if worm_burden.size == 0:
            return 0.0
        else:
            return float(np.mean(worm_burden > 0))

    def calc_proportion_worms(self, numerator="female", denom="all") -> float:
        """
        Calculates the proportion of worm type A to type(s) B in the entire host population.

        @param numerator: can be one of "female", "male", or fertile_female". Denotes which type of worm
            will be used in the numerator of the proportion.
        @param denom: can be one of "all", "female", "male", or fertile_female". Denotes which type of worm
            will be used in the denominator of the proportion. "all" will use all worm types.

        Returns:
            float: Ratio of total worms of type numerator to type denominator
        """
        numerator_burden = self.worm_burden_per_person(worm_type=numerator)
        if numerator_burden.size == 0:
            return 0.0
        return float(
            (numerator_burden.sum()) /
            self.worm_burden_per_person(worm_type=denom).sum()
        )

    def calculate_l3_per_blackfly(self) -> float:
        if len(self.people.blackfly.L3) == 0:
            return 0.0
        return np.mean(self.people.blackfly.L3)

    def calculate_atp(self) -> float:
        return self.calculate_l3_per_blackfly() * self._params.blackfly.bite_rate_per_person_per_year

    def calculate_prevalence_l3_blackflies(self) -> float:
        l3_intensity = self.calculate_l3_per_blackfly()
        k = l3_intensity * self._params.blackfly.k1 + self._params.blackfly.k0
        prevalence = 1 - (1 + l3_intensity / k)**(-k)
        return prevalence

    def _update_for_epilepsy(self):
        current_test_for_OAE = self.people.get_current_tested_for_OAE()
        if current_test_for_OAE.sum() > 0:
            self.people.tested_for_OAE |= current_test_for_OAE
            _, measured_mf = self.microfilariae_per_skin_snip()
            rounded_mf: Array.Person.Int = np.round(measured_mf[current_test_for_OAE])
            epilepsy_prob = self.fit_func_OAE(rounded_mf)
            out = np.equal(self.derived_params.numpy_bit_generator.binomial(1, epilepsy_prob), 1)
            self.people.has_OAE[current_test_for_OAE] |= out

    def OAE_prevalence(self) -> float:
        return self.people.has_OAE.sum() / self.n_people

    def sequalae_prevalence(self) -> dict[str, float]:
        sequelae_prevalence = {}

        if "APOD" in self.people.has_sequela and "CPOD" in self.people.has_sequela:
            apod = self.people.has_sequela["APOD"]
            cpod = self.people.has_sequela["CPOD"]
            has_rsd = np.logical_or(apod, cpod)
            sequelae_prevalence["RSDComplex"] = has_rsd.sum() / self.n_people

        for name, sequela in self.people.has_sequela.items():
            prev = sequela.sum() / self.n_people
            if name == "Blindness":
                visual_prev = prev * 1.78
                if visual_prev > 1:
                    new_visual_prev = 1
                else:
                    new_visual_prev = visual_prev
                sequelae_prevalence["Blindness"] = prev
                sequelae_prevalence["VisualImpairment"] = new_visual_prev
            elif name == "RSD":
                sequelae_prevalence["RSDSimple"] = prev
            else:
                sequelae_prevalence[name] = prev
        return sequelae_prevalence
    
    def sample_seroprevalence(self, sens_spec: tuple[float, float], seroreversion=False) -> float:
        """
        Calculates the the seroprevalence in the population using the given sensitivity and specificity values.
        Seroreversion in the absence of infection is only assumed when set to True.

        @param sens_spec: a tuple containing (Sens, Spec) of the diagnostic test, of the range [0, 1].
        @param seroreversion: a boolean, which if set to true outputs seroprevalence using seroreversion.

        Returns:
            float: seroprevalence in population given supplied sens/spec (and seroreversion if specified)
        """
        serostatus = self.people.ov16_serostatus
        ov16_pos_mask = np.where(serostatus == True)[0]
        ov16_neg_mask = np.where(serostatus == False)[0]
        sampled_serostatus = np.zeros(self.n_people)
        sampled_serostatus[ov16_pos_mask] = self.people.ov16_diagnostic_rand[ov16_pos_mask] <= sens_spec[0]
        sampled_serostatus[ov16_neg_mask] = self.people.ov16_diagnostic_rand[ov16_neg_mask] > sens_spec[1]
        return np.mean(sampled_serostatus)

    def percent_non_compliant(self) -> float:
        min_age = (
            SpecificTreatmentParams().min_age_of_treatment
            if (self._params.treatment is None)
            else self._params.treatment.min_age_of_treatment
        )
        eligible_population = self.people.ages >= min_age
        if np.sum(eligible_population) > 0:
            return 1 - np.mean(self.people.has_been_treated[eligible_population])
        return 1


def make_state_from_params(params: Params):
    return State.from_params(params)


def make_state_from_hdf5(input_file: str | Path | IO[bytes]):
    return State.from_hdf5(input_file)
