import warnings
from enum import Enum
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from itertools import groupby
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from utilities.util import make_fixed_sticky_tpm_JAX, make_sample_weights_which_mask_the_initial_timestep_for_each_event, example_end_times_are_proper
from utilities.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
)

from model import Model 
from params import (
    AllParameters_JAX,
    ContinuousStateParameters_JAX,
    Dims,
    EntityTransitionParameters_MetaSwitch_JAX,
    InitializationParameters_JAX,
    SystemTransitionParameters_JAX,
)
from expectation_step import run_VES_step_JAX, run_VEZ_step_JAX
from maximization_step import (
    M_Step_Toggle_Value,
    run_M_step_for_CSP_in_closed_form__Gaussian_case,
    run_M_step_for_ETP_via_gradient_descent,
    run_M_step_for_IP,
    run_M_step_for_STP_in_closed_form,
    run_M_step_for_STP_via_gradient_descent,
    compute_STP_closed_form_M_step,
    compute_ETP_closed_form_M_step_on_posterior_summaries
)
from compute_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX
)


"""
Computes the model initialization strategy. 
"""

@dataclass
class InitializationResults:

    """
    Purpose: defines the variables necessary for saving all of the initialization results. 

    Attributes: 
        params: STP, ETP, CSP, IP
        ES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
                given the entire observation sequence, and the probability density over emissions.  
        EZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
                given the entire observation sequence, and the probability density over the observations. 
        record_of_most_likely_system_states: argmax(VES_summary.expected_regimes) over all time steps T across N examples 
        record_of_most_likely_entity_states: argmax(VEZ_summary.expected_regimes) over all time steps T across N examples foe each of the J entities   
    """

    params: AllParameters_JAX
    ES_summary: HMM_Posterior_Summary_JAX
    EZ_summaries: HMM_Posterior_Summaries_JAX
    record_of_most_likely_system_states: NumpyArray2D  # Txnum_EM_iterations
    record_of_most_likely_entity_states: NumpyArray3D  # TxJx num_EM_iterations


@dataclass
class ResultsFromBottomHalfInit:
    """
    Purpose: defines the variables necessary for saving the results from the bottom half - i.e. the VEZ summary (maringals, pairwise marginals, emission density),
    the continuous state and entity state MODEL paremeters.  

    Attributes: 
        CSP: the observation parameters A[j,k], b[j,k], Q[j,k]
        EZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        record_of_most_likely_states: argmax(VEZ_summary.expected_regimes) over all time steps T across N examples foe each of the J entities  
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K) 
    """

    CSP: ContinuousStateParameters_JAX
    EZ_summaries: HMM_Posterior_Summaries_JAX
    record_of_most_likely_states: NumpyArray3D  # TxJx num_EM_iterations
    ETP: Optional[EntityTransitionParameters_MetaSwitch_JAX] = None


@dataclass
class ResultsFromTopHalfInit:
    """
    Purpose: defines the variables necessary for saving the results from the top half - i.e. the VES summary (maringals, pairwise marginals, emission density),
    the system and entity state MODEL paremeters.  

    Attributes: 
        STP: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K) 
        ES_summaries: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions.  
        record_of_most_likely_states: argmax(VES_summary.expected_regimes) over all time steps T across N examples 
    """


    STP: SystemTransitionParameters_JAX
    ETP: EntityTransitionParameters_MetaSwitch_JAX
    ES_summary: HMM_Posterior_Summary_JAX
    record_of_most_likely_states: NumpyArray2D  # Txnum_EM_iterations


@dataclass
class RawInitializationResults:
    """
    Purpose: Compared to `InitializationResults`, this representation is closer to how the initialization was constructed:
    there's info from a "bottom-level" AR-HMM and from a "top-level" AR-HMM. These results are also useful for inspecting the quality of the initialization
    (e.g. via `top.record_of_most_likely_states` or `bottom.record_of_most_likely_states`)
    with respect to known truth.

    Attributes: 
        bottom: CSP parms, EZ_summaries, record_of_most_likely_states, ETP params
        top: STP parms, ES_summaries, record_of_most_likely_states, ETP params
        IP: the initial emission parameters pi_system, pi_entities, mu_0s, Sigma_0s
    """

    bottom: ResultsFromBottomHalfInit
    top: ResultsFromTopHalfInit
    IP: InitializationParameters_JAX


def initialization_results_from_raw_initialization_results(
    raw_initialization_results: RawInitializationResults,
    params_frozen: Optional[AllParameters_JAX] = None,
):
    """
    Purpose: Return the full initialization results from the "raw" initialization results from the top and bottom half.

    Attributes: 
        raw_initialization_results: the specific summaries and parameters from from the top and bottom half pre-training 
        params_frozen: frozen version of all params STP, ETP, CSP, IP

    Returns: 
        Initialization results (VES summary, VEZ summary, most likely states for system (top), most likely states for bottom (entity))
    """
    RI = raw_initialization_results
    if params_frozen:
        params = params_frozen
    else:
        params = AllParameters_JAX(RI.top.STP, RI.top.ETP, RI.bottom.CSP, RI.IP)
    return InitializationResults(
        params,
        RI.top.ES_summary,
        RI.bottom.EZ_summaries,
        RI.top.record_of_most_likely_states,
        RI.bottom.record_of_most_likely_states,
    )


def make_data_free_preinitialization_of_IP_JAX(DIMS, shared_variance=1.0) -> InitializationParameters_JAX:
    """
    Purpose: Return the initialization parameters without any influence from data. All uniform initial parameters. 

    Attributes: 
        DIMS: list of amounts of all parameters 
        shared_variance: the variance of the observation gaussian distributions for each entity 

    Returns: 
        IP: the initial emission parameters pi_system, pi_entities, mu_0s, Sigma_0s
    """
    pi_system = np.ones(DIMS.L) / DIMS.L
    pi_entities = np.ones((DIMS.J, DIMS.K)) / DIMS.K
    mu_0s = jnp.zeros((DIMS.J, DIMS.K, DIMS.D))
    Sigma_0s = jnp.tile(shared_variance * jnp.eye(DIMS.D), (DIMS.J, DIMS.K, 1, 1))
    return InitializationParameters_JAX(pi_system, pi_entities, mu_0s, Sigma_0s)


def make_tpm_only_preinitialization_of_STP_JAX(
    DIMS: Dims, fixed_self_transition_prob: float
) -> SystemTransitionParameters_JAX:
    """
    Purpose: Return the transition probability parameters for the system states without any influence from data. Sample from a Sticky Dirichlet prior
    to obtain the Pi parameters across time, and take the log. Set Upsilon parameters to all zeros.  

    Attributes: 
        DIMS: list of amounts of all parameters 
        fixed_self_transition_prob: probability of remaining in the current state at the next time-step

    Returns: 
        STP: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)
    """
   
    L, J, K, D_s = DIMS.L, DIMS.J, DIMS.K, DIMS.D_s
    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=L)
    Pi = jnp.log(tpm)
    Upsilon = jnp.zeros((L, D_s))
    return SystemTransitionParameters_JAX(Upsilon, Pi)


def make_data_free_preinitialization_of_STP_JAX(
    DIMS: Dims,
    method_for_Upsilon: str,
    fixed_self_transition_prob: float,
    seed: int,
) -> SystemTransitionParameters_JAX:
    """
    Purpose: Return the transition probability parameters for the system states without any influence from data. Sample from a Sticky Dirichlet prior
    to obtain the Pi parameters across time, and then take the log. The Upsilon parameters can be all zeros or can be sampled from a normal distribution. 

    Attributes: 
        DIMS: list of amounts of all parameters 
        method_for_Upsilon: str to indicate the initialization method for Upsilon (e.g. "zeros")
        fixed_self_transition_prob: probability of remaining in the current state at the next time-step
        seed: Randomized seed for Upsilon parameters 

    Returns: 
        STP: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)
    """
    key = jr.PRNGKey(seed)
    # TODO: Support fixed or random draws from prior.
    L, J, K, D_s = DIMS.L, DIMS.J, DIMS.K, DIMS.D_s


    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=L)
    Pi = jnp.log(tpm)

    if method_for_Upsilon == "rnorm":
        Upsilon = jr.normal(key, (L, D_s))
    elif method_for_Upsilon == "zeros":
        Upsilon = jnp.zeros((L, D_s))
    else:
        raise ValueError("What is the method for Upsilon?")
    return SystemTransitionParameters_JAX(Upsilon, Pi)


def make_data_free_preinitialization_of_ETP_JAX(
    DIMS: Dims,
    method_for_Psis: str,
    seed: int,
    fixed_self_transition_prob: float = 0.90,
) -> EntityTransitionParameters_MetaSwitch_JAX:
    """
    Purpose: Return the transition probability parameters for the entity states without any influence from data.
    Sample from a Sticky Dirichlet prior to obtain the Ps parameters across time, and then take the log.
    The Psis parameters can be all zeros or can be sampled from a normal distribution. 

    Attributes: 
        DIMS: list of amounts of all parameters 
        method_for_Upsilon: str to indicate the initialization method for Upsilon (e.g. "zeros")
        seed: Randomized seed for Psis parameters 
        fixed_self_transition_prob: probability of remaining in the current state at the next time-step

    Returns: 
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
    """
    key = jr.PRNGKey(seed)
    L, J, K, D_e = DIMS.L, DIMS.J, DIMS.K, DIMS.D_e
    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=K)
    Ps = jnp.tile(np.log(tpm), (J, L, 1, 1))
    if method_for_Psis == "rnorm":
        Psis = jr.normal(key, (J, L, K, D_e))
    elif method_for_Psis == "zeros":
        Psis = jnp.zeros((J, L, K, D_e))
    else:
        raise ValueError("What is the method for Psis?")
    return EntityTransitionParameters_MetaSwitch_JAX(Psis,Ps)


def make_tpm_only_preinitialization_of_ETP_JAX(
    DIMS: Dims, fixed_self_transition_prob: float
) -> EntityTransitionParameters_MetaSwitch_JAX:
    """
    Purpose: Return the transition probability parameters for the entity states without any influence from data.
    Sample from a Sticky Dirichlet prior to obtain the Ps parameters across time, and then take the log.

    Attributes: 
        DIMS: list of amounts of all parameters 
        fixed_self_transition_prob: probability of remaining in the current state at the next time-step

    Returns: 
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
    """
    L, J, K, D_e = DIMS.L, DIMS.J, DIMS.K, DIMS.D_e
    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=K)
    Ps = jnp.tile(np.log(tpm), (J, L, 1, 1))
    Psis = jnp.zeros((J, L, K, D_e))
    return EntityTransitionParameters_MetaSwitch_JAX(Psis, Ps)



def make_kmeans_preinitialization_of_CSP_JAX(
    DIMS: Dims,
    observations: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_observations: Optional[JaxNumpyArray2D] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    plotbose: bool = False,
) -> Tuple[ContinuousStateParameters_JAX, sklearn.cluster._kmeans.KMeans]:
    """
    Purpose: Initizlize the emission parameters CSP by assigning the observations to K regimes by applying k-means to the values of the observations. 
    We then initialize CSP parameters by running separate vector autoregressions within each cluster/regime:
        - We find regime-specific state matrix (CSP.As) and biases (CSP.bs) by applying a (multi-outcome) linear regression
            to predict the next observation from the previous observation.
        - We estimate the regime-specific covariance matrices (CSP.Qs) from the residuals of the above linear regresssion.
    Note that all x_t^j from a single k regime are predicted from its x_{t-1}^j that can be from another k cluster. 

    Arguments:
        DIMS: list of amounts of all parameters 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        save_dir: str for the path to save the file
        verbose: True boolean if we want to print update statements during training 
        plotbose: verbose in plotting

    Return: Initialized CSP parameters: A[j,k], b[j,k], Q[j,k]
    """
    if verbose:
        print("Now performing k-means pre-initialization of CSP parameters.")

    ### Up-front computations
    observations = jnp.asarray(observations)
    T, J, D = np.shape(observations)
    K = DIMS.K

    ### Make sample weights (as a combo of `mask_observations`` and `example_end_times`)
    sample_weights = make_sample_weights_which_mask_the_initial_timestep_for_each_event(
        observations,
        example_end_times,
        mask_observations,
    )

    As = np.zeros((J, K, D, D))
    bs = np.zeros((J, K, D))
    Qs = np.tile(np.eye(D)[None, None, :, :], (J, K, 1, 1))

    continuous_state_diffs = observations[1:, :, :] - observations[:-1, :, :]

    data_for_kmeans = observations
    weights_for_kmeans = sample_weights


    kms = [None] * J
    for j in range(J):
        with warnings.catch_warnings():
            # sklearn gives the warning below, but I don't know how to execute on the advice, currently.
            #   FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4.
            #   Set the value of `n_init` explicitly to suppress the warning
            warnings.simplefilter(action="ignore", category=FutureWarning)
            kms[j] = KMeans(K, random_state=120).fit(data_for_kmeans[:, j, :], sample_weight=weights_for_kmeans[:, j])

    for j in range(J):
        for k in range(K):
            ### find which samples to use
            # we only use samples which are in the cluster currently under consideration IF they got weighted
            # non-zero in the k-means calculation
            samples_are_in_cluster_jk = kms[j].labels_ == k
            bools_use_pair_for_cluster_jk = samples_are_in_cluster_jk * weights_for_kmeans[:, j]

            outcome_indices_jk = np.where(bools_use_pair_for_cluster_jk)[0]
            predictor_indices_jk = outcome_indices_jk - 1

            outcomes_jk = observations[outcome_indices_jk, j, :]
            predictors_jk = observations[predictor_indices_jk, j, :]
            ### run vector autoregression
            lr = LinearRegression(fit_intercept=True)
            lr.fit(predictors_jk, outcomes_jk)
            As[j, k] = lr.coef_ 
            bs[j, k] = lr.intercept_
            expectations_jk = (As[j, k] @ predictors_jk.T).T + bs[j, k]
            residuals_jk = outcomes_jk - expectations_jk
            # tied_residuals_j = np.concatenate((tied_residuals_j, residuals_jk))
            Qs[j, k] = np.cov(residuals_jk, rowvar=False)

    As = jnp.asarray(As)
    bs = jnp.asarray(bs)
    Qs = jnp.asarray(Qs)
    return ContinuousStateParameters_JAX(As, bs, Qs), kms


###
# AR-HMM (BOTTOM  HALF)
###


def fit_rARHMM_to_bottom_half_of_model(
    observations: JaxNumpyArray3D,
    example_end_times: Optional[JaxNumpyArray1D],
    CSP_JAX: ContinuousStateParameters_JAX,
    ETP_JAX: EntityTransitionParameters_MetaSwitch_JAX,
    IP_JAX: InitializationParameters_JAX,
    model: Model,
    num_EM_iterations: int,
    treat_ETP_params_as_tpm: bool = False,
    mask_observations: Optional[JaxNumpyArray2D] = None,
    params_frozen: Optional[AllParameters_JAX] = None,
    outside_entity_recurrence: Optional[JaxNumpyArray3D] = None,
    verbose: bool = True,
) -> ResultsFromBottomHalfInit:
    """
    Purpose: Initizlize the bottom half of the HSRDM - meaning the CSP and the ETP parameters. The initialized CSP, ETP, and IP 
    parameters are instantiated. Then we start with a uniform expected VES regimes and take the expectation of the MODEL log entity transition probabilities with respect to the VES posterior. 
    Then one computes forwards-backwards to obtain the 
    VEZ posterior. The MODEL parameters are then updated via an M-step. 

    Arguments:
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        CSP_JAX: the observation parameters A[j,k], b[j,k], Q[j,k]
        ETP_JAX: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        IP_JAX: the initial emission parameters pi_system, pi_entities, mu_0s, Sigma_0s
        model: joint distribution defined in -> (model.py)
        num_EM_iterations: total number of EM iterations
        treat_ETP_params_as_tpm: Boolean to treat ETP parameters as a tmp; has to do with doing a closed_form M step for ETP parameters. Set to False. 
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        params_frozen: frozen version of all params STP, ETP, CSP, IP
        outside_entity_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        verbose: True boolean if we want to print update statements during training 

    Return: Results from bottom initialization: CSP params, EZ_summaries, ETP params, most likely entity states  
    """

    T = len(observations)
    J, L, K, _ = np.shape(ETP_JAX.Ps)

    record_of_most_likely_states = np.zeros((T, J, num_EM_iterations), dtype=int)

    if verbose:
        print("\n--- Now running AR-HMM on bottom half of Model 2a. ---")
    for i in range(num_EM_iterations):
        if verbose:
            print(f"Now running EM iteration {i+1}/{num_EM_iterations} for AR-HMM on bottom half of Model 2a.")

        ###
        # E-step
        ###

        VES_expected_regimes__uniform = np.ones((T, L)) / L

        EZ_summaries = run_VEZ_step_JAX(
            CSP_JAX,
            ETP_JAX,
            IP_JAX,
            observations,
            VES_expected_regimes__uniform, 
            model,
            example_end_times,
            outside_entity_recurrence
        )

        for j in range(J):
            record_of_most_likely_states[:, j, i] = np.array(
                np.argmax(EZ_summaries.expected_regimes[:, j, :], axis=1), dtype=int
            )

        ###
        # M-step (for ETP, represented as a transition probability matrix)
        ###

        if params_frozen:
            ETP_JAX = params_frozen.ETP
            CSP_JAX = params_frozen.CSP
        else:
            ###
            # M-step (for ETP)
            ###

            if treat_ETP_params_as_tpm:
                # We need it to have shape (J,L,K,K).  So just do it with (J,K,K), then tile it over L.
                # TODO: This is rewriting the logic of "compute_closed_form_M_step."  Be sure that that can
                # work when we have J tpms, and then
                tpms = compute_ETP_closed_form_M_step_on_posterior_summaries(
                    EZ_summaries,
                    mask_observations,
                    example_end_times,
                )
                Ps_new = jnp.tile(jnp.log(tpms[:, None, :, :]), (1, L, 1, 1))
                ETP_JAX = EntityTransitionParameters_MetaSwitch_JAX(ETP_JAX.Psis, Ps_new)

            else:
                num_M_step_iterations_for_ETP_gradient_descent = 5
                ES_summary_uniform = HMM_Posterior_Summary_JAX(
                    expected_regimes=VES_expected_regimes__uniform,  
                    expected_joints=jnp.ones((T - 1, L, L)) / L,
                    log_normalizer=jnp.nan,
                )
                ETP_JAX = run_M_step_for_ETP_via_gradient_descent(
                    ETP_JAX,
                    ES_summary_uniform,
                    EZ_summaries,
                    observations,
                    i,
                    num_M_step_iterations_for_ETP_gradient_descent,
                    model,
                    example_end_times,
                    outside_entity_recurrence,
                    mask_observations,
                    verbose,
                )

            ###
            # # M-step (for CSP)
            # ###
            CSP_JAX = run_M_step_for_CSP_in_closed_form__Gaussian_case(
                EZ_summaries.expected_regimes,
                observations,
                example_end_times,
                mask_observations,
            )
    return ResultsFromBottomHalfInit(CSP_JAX, EZ_summaries, record_of_most_likely_states, ETP_JAX)


###
# AR-HMM (TOP HALF)
###


def fit_ARHMM_to_top_half_of_model(
    observations: NumpyArray3D,
    example_end_times: Optional[JaxNumpyArray1D],
    STP_JAX: SystemTransitionParameters_JAX,
    ETP_JAX: EntityTransitionParameters_MetaSwitch_JAX,
    IP_JAX: InitializationParameters_JAX,
    EZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    num_EM_iterations: int,
    num_M_step_iterations_for_ETP_gradient_descent: int,
    mask_observations: Optional[JaxNumpyArray2D] = None,
    params_frozen: Optional[AllParameters_JAX] = None,
    outside_system_recurrence: Optional[JaxNumpyArray2D] = None,
    outside_entity_recurrence: Optional[JaxNumpyArray3D] = None,
    verbose: bool = True,
) -> ResultsFromTopHalfInit:

    """
    Purpose: Initizlize the top half of the HSRDM - meaning the STP parameters. The initialized STP parameters are instantiated. 
    Then we start with the current VEZ regimes and take the expectation of the MODEL log system transition probabilities with respect to the VEZ posterior. 
    Then one computes forwards-backwards to obtain the new VES posterior. The MODEL parameters for STP and ETP are then updated via an M-step. 

    Arguments:
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        STP_JAX: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)
        ETP_JAX: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        IP_JAX: the initial emission parameters pi_system, pi_entities, mu_0s, Sigma_0s
        EZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        num_EM_iterations: total number of EM iterations
        num_M_step_iterations_for_ETP_gradient_descent: number of iterations for ETP gradient descent 
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        params_frozen: frozen version of all params STP, ETP, CSP, IP
        outside_system_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        outside_entity_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        verbose: True boolean if we want to print update statements during training 

    Return: Results from bottom initialization: CSP params, EZ_summaries, ETP params, most likely entity states  
    """
   
    T = len(observations)
    record_of_most_likely_states = np.zeros((T, num_EM_iterations))


    if verbose:
        print("\n--- Now running AR-HMM on top half of Model 2a. ---")

    for iteration in range(num_EM_iterations):
        if verbose:
            print(f"Now running EM iteration {iteration+1}/{num_EM_iterations} for AR-HMM on top half of Model 2a.")

        ###
        # E-step
        ###
        ES_summary = run_VES_step_JAX(
            STP_JAX,
            ETP_JAX,
            IP_JAX,
            observations,
            EZ_summaries,
            model,
            example_end_times,
            outside_system_recurrence,
            outside_entity_recurrence,
            mask_observations=mask_observations,
        )
        

        record_of_most_likely_states[:, iteration] = np.array(np.argmax(ES_summary.expected_regimes, axis=1), dtype=int)

        ###
        # M-step
        ###

        if params_frozen:
            ETP_JAX = params_frozen.ETP
            STP_JAX = params_frozen.STP
        else:
            ### M-step (ETP)
            ETP_JAX = run_M_step_for_ETP_via_gradient_descent(
                ETP_JAX,
                ES_summary,
                EZ_summaries,
                observations,
                iteration,
                num_M_step_iterations_for_ETP_gradient_descent,
                model,
                example_end_times,
                outside_entity_recurrence,
                mask_observations,
                verbose,
            )

            system_transition_prior = None

            ### M-step (STP)
            num_system_states = np.shape(STP_JAX.Pi)[0]
            if num_system_states == 1:
                # TODO: I had written earlier that the VES step has already taken care of the `mask_observations` mask.
                # But I might want to double check that.
                STP_JAX = run_M_step_for_STP_in_closed_form(STP_JAX, ES_summary, example_end_times)
            else:
                NUM_M_STEP_ITERATIONS_FOR_STP_GRADIENT_DESCENT = 5
                STP_JAX = run_M_step_for_STP_via_gradient_descent(
                    STP_JAX,
                    ES_summary,
                    system_transition_prior,
                    iteration,
                    NUM_M_STEP_ITERATIONS_FOR_STP_GRADIENT_DESCENT,
                    model,
                    example_end_times,
                    outside_system_recurrence,
                    observations,
                    verbose,
                )
    
    return ResultsFromTopHalfInit(STP_JAX, ETP_JAX, ES_summary, record_of_most_likely_states)


###
# MAIN
###


def initialize_HSRDM(
    DIMS: Dims,
    observations: Union[NumpyArray3D, JaxNumpyArray3D],
    example_end_times: Optional[NumpyArray1D],
    model: Model,
    num_em_iterations_for_bottom_half: int = 5,
    num_em_iterations_for_top_half: int = 20,
    seed: int = 120,
    mask_observations: Optional[JaxNumpyArray2D] = None,
    save_dir: Optional[str] = None,
    treat_ETP_params_as_tpm_during_bottom_half_inference: bool = True,
    params_frozen: Optional[AllParameters_JAX] = None,
    outside_system_recurrence: Optional[JaxNumpyArray2D] = None,
    outside_entity_recurrence: Optional[JaxNumpyArray3D] = None,
    verbose: bool = True,
    plotbose: bool = False,
) -> InitializationResults:
    """
    Purpose: Initializes the STP, ETP, CSP and IP parameters of the HSRDM. The STP params are initialized as a sticky transition matrix and sampled from a random normal for the recurrence. 
        The ETP params are initialized as a sticky transition matrices and sampled from a random normal for the recurrence. The IP params are initialized with 
        a uniform categorical distribution for the latent state and the continuous states having zero mean with an identity isotropic covariance. This is all a "data free" initialization.
        For the CSP parameters, we currently initialize with a k-means scheme. Could make this data free in the future.
    Arguments:
        DIMS: list of amounts of all parameters 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        model: joint distribution defined in -> (model.py)
        num_EM_iterations_for_bottom_half: total number of EM iterations for the bottom 
        num_EM_iterations_for_top_half: total number of EM iterations for the top
        seed: for random initializations 
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        save_dir: str for the path to save the file
        treat_ETP_params_as_tpm_during_bottom_half_inference: Boolean to treat ETP parameters as a tmp; has to do with doing a closed_form M step for ETP parameters. Set to True
        as in the bottom half, the ETP is like the "system state". 
        params_frozen: frozen version of all params STP, ETP, CSP, IP
        outside_system_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        outside_entity_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        verbose: True boolean if we want to print update statements during training 
        plotbose: Verbose in plotting 

    Return: Initialization results. 
    """
 
    if example_end_times is None:
        T = len(observations)
        example_end_times = np.array([-1, T])

    if not example_end_times_are_proper(example_end_times, len(observations)):
        raise ValueError(
            f"Event end times do not have the proper format. Consult the `events` module "
            f"and try again.  Event_end_times MUST begin with -1 and end with T, the length "
            f"of the grand time series."
        )

    observations = jnp.asarray(observations)

    ###
    # Initialize Params
    ####

    if params_frozen:
        CSP_JAX = params_frozen.CSP
        ETP_JAX = params_frozen.ETP
        IP_JAX = params_frozen.IP

    else:
        CSP_JAX, kms = make_kmeans_preinitialization_of_CSP_JAX(
            DIMS,
            observations,
            example_end_times,
            mask_observations,
            save_dir,
            verbose,
            plotbose,
        )
        ETP_JAX = make_data_free_preinitialization_of_ETP_JAX(
            DIMS, method_for_Psis="rnorm", fixed_self_transition_prob=0.90, seed=seed)
        IP_JAX = make_data_free_preinitialization_of_IP_JAX(DIMS)


    ###
    # Fit Bottom-level HMM
    ###
    results_bottom = fit_rARHMM_to_bottom_half_of_model(
        observations,
        example_end_times,
        CSP_JAX,
        ETP_JAX,
        IP_JAX,
        model,
        num_em_iterations_for_bottom_half,
        treat_ETP_params_as_tpm_during_bottom_half_inference,
        mask_observations,
        params_frozen,
        outside_entity_recurrence,
        verbose,
    )

    ###
    # Top-level HMM
    ###

    if params_frozen:
        STP_JAX = params_frozen.STP
    else:
        ### Initialization
        STP_JAX = make_data_free_preinitialization_of_STP_JAX(
            DIMS,
            method_for_Upsilon="rnorm",
            fixed_self_transition_prob=0.95,
            seed=seed,
        )
  
    ### run HMM
    num_M_step_iterations_for_ETP_gradient_descent = 5


    results_top = fit_ARHMM_to_top_half_of_model(
        observations,
        example_end_times,
        STP_JAX,
        results_bottom.ETP,
        IP_JAX,
        results_bottom.EZ_summaries,
        model,
        num_em_iterations_for_top_half,
        num_M_step_iterations_for_ETP_gradient_descent,
        mask_observations,
        params_frozen,
        outside_system_recurrence,
        outside_entity_recurrence,
        verbose=verbose,
    )

    ### Update Initialization Params
    if params_frozen:
        IP_JAX = params_frozen.IP
    else:
        IP_JAX = run_M_step_for_IP(
            IP_JAX,
            M_Step_Toggle_Value.CLOSED_FORM_GAUSSIAN,
            results_top.ES_summary,
            results_bottom.EZ_summaries,
            observations,
            example_end_times,
        )


    results_raw = RawInitializationResults(results_bottom, results_top, IP_JAX)
    return initialization_results_from_raw_initialization_results(results_raw, params_frozen)
