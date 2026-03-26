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
from sklearn.linear_model import Ridge
import pandas as pd
from pathlib import Path
import json 
import os

from utilities.util import make_fixed_sticky_tpm_JAX, make_sample_weights_which_mask_the_initial_timestep_for_each_event, example_end_times_are_proper, generate_silent_observation, clamp_parameters, extract_pooled_ar_pairs_by_evidence_with_silent_predictor, mean_gaussian_loglik_ar, extract_pooled_ar_pairs_to_silence
from utilities.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray4D,
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


def make_data_free_preinitialization_of_IP_JAX(DIMS, observations, shared_variance=1.0) -> InitializationParameters_JAX:
    """
    Purpose: Return the initialization parameters without any influence from data. All uniform initial parameters. 

    Attributes: 
        DIMS: list of amounts of all parameters 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        shared_variance: the variance of the observation gaussian distributions for each entity 

    Returns: 
        IP: the initial emission parameters pi_system, pi_entities, mu_0s, Sigma_0s
    """
    pi_system = np.ones(DIMS.L) / DIMS.L
    pi_entities = np.ones((DIMS.J, DIMS.K)) / DIMS.K
    
    K_SPECIAL = 0 
    fixed_means_per_entity, fixed_cov = generate_silent_observation(observations)
    mu_0s = jnp.zeros((DIMS.J, DIMS.K, DIMS.D))
    mu_0s = mu_0s.at[:, K_SPECIAL, :].set(fixed_means_per_entity)

    Sigma_0s = jnp.tile(shared_variance * jnp.eye(DIMS.D), (DIMS.J, DIMS.K, 1, 1))
    Sigma_0s = Sigma_0s.at[:, K_SPECIAL, :, :].set(fixed_cov)

    K_SPECIAL = 1
    mu_0s = mu_0s.at[:, K_SPECIAL, :].set(fixed_means_per_entity)
    Sigma_0s = Sigma_0s.at[:, K_SPECIAL, :, :].set(fixed_cov)

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
    hard_code_scale: float = 2.0
) -> SystemTransitionParameters_JAX:
    """
    Purpose: Return the transition probability parameters for the system states without any influence from data. Sample from a Sticky Dirichlet prior
    to obtain the Pi parameters across time, and then take the log. The Upsilon parameters can be all zeros or can be sampled from a normal distribution. 

    Attributes: 
        DIMS: list of amounts of all parameters 
        method_for_Upsilon: str to indicate the initialization method for Upsilon (e.g. "zeros")
        fixed_self_transition_prob: probability of remaining in the current state at the next time-step
        seed: Randomized seed for Upsilon parameters 
        hard_code_scale: the weightings for scaling the transitions between L=0,1 (only for when L=2)
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
    elif method_for_Upsilon == "hard_code":
        Upsilon = jnp.zeros((L, D_s)) 
        # Row 0: [2, 0, 0, ..., 0]
        ramp2 = 1 * (jnp.arange(D_s-1) + 2)  # length D_s-1
        Upsilon = Upsilon.at[0, 0].set(hard_code_scale)
        Upsilon = Upsilon.at[1, 1:].set(ramp2)

            
    else:
        raise ValueError("What is the method for Upsilon?")

    return SystemTransitionParameters_JAX(Upsilon, Pi)



def make_data_free_preinitialization_of_ETP_JAX(
    DIMS: Dims,
    method_for_Psis: str,
    seed: int,
    fixed_self_transition_prob: float = 0.25,
    hard_code_scale_no_evidence: float = 2.0,
) -> EntityTransitionParameters_MetaSwitch_JAX:
    """
    Purpose: Return the transition probability parameters for the entity states without any influence from data.
    Sample from a Sticky Dirichlet prior to obtain the Ps parameters across time, and then take the log.
    The Psis parameters can be all zeros, can be sampled from a normal distribution, or can be a hard coded matrix that is the same 
    for all entities. 

    Attributes: 
        DIMS: list of amounts of all parameters 
        method_for_Upsilon: str to indicate the initialization method for Upsilon (e.g. "zeros")
        seed: Randomized seed for Psis parameters 
        fixed_self_transition_prob: probability of remaining in the current state at the next time-step
        hard_code_scale_no_evidence: the weightings for scaling the entity transitions to K=0,2 when L=0
        hard_code_scale_ramp: the ramp weightings for scaling the entity transitions to K=1,3 when L=1

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
    elif method_for_Psis == "hard_code":
        Psis_shared = jnp.zeros((L, K, D_e))

        # --- L index mapping ---
        # Let's say L=0 -> "no evidence / feedback off"
        # and L=1 -> "evidence / feedback on"
        L_no = 0
        L_yes = 1

        # --- No evidence: boost destination states 0 and 2 when evidence class 0 is high ---
        # Equivalent to: b_k = 2*e0 for k in {0,2}
        Psis_shared = Psis_shared.at[L_no, 0, 0].set(hard_code_scale_no_evidence)
        Psis_shared = Psis_shared.at[L_no, 2, 0].set(2) 

        # --- Evidence present: ramp increases with evidence class for destination state 2 (edit if needed) ---
        # ramp[c] = 0.5 * c  (with ramp[0]=0)
        ramp2 = (jnp.arange(D_e) + 2)  # length D_e-1
        Psis_shared = Psis_shared.at[L_yes, 1, :].set(ramp2)
        Psis_shared = Psis_shared.at[L_yes, 3, 0].set(2) 


        # Tile across entities
        Psis = jnp.broadcast_to(Psis_shared, (J, L, K, D_e))     
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
    seed: int = 126, 
    save_dir: Optional[str] = None,
    verbose: bool = True,
    share_params_across_entities: bool = True,
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
        seed: random_state for k-means algorithm 
        save_dir: str for the path to save the file
        verbose: True boolean if we want to print update statements during training 
        share_params_across_entities: boolean = True if you plan to share the params across the entities; false otherwise 

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

    if not share_params_across_entities:

        kms = [None] * J
        for j in range(J):
            with warnings.catch_warnings():
                # sklearn gives the warning below, but I don't know how to execute on the advice, currently.
                #   FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4.
                #   Set the value of `n_init` explicitly to suppress the warning
                warnings.simplefilter(action="ignore", category=FutureWarning)
                kms[j] = KMeans(K, random_state=seed).fit(data_for_kmeans[:, j, :], sample_weight=weights_for_kmeans[:, j]) #initial cluster assignment is randomized by random_state seed 

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
                Q = np.cov(residuals_jk, rowvar=False)
                Q = Q + 1e-4 * np.eye(D) #Floors the diagonal variances to 1e^-4 so that they can never get too small 
                Qs[j, k] = Q

    else: 
        X_km = data_for_kmeans.reshape(T * J, D)
        w_km = weights_for_kmeans.reshape(T * J)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            km_shared = KMeans(K, random_state=seed).fit(X_km, sample_weight=w_km)

        labels_TJ = km_shared.labels_.reshape(T, J)

        kms = [km_shared] * J

        for k in range(K):
            in_k = (labels_TJ == k)
            ok = in_k & (weights_for_kmeans > 0)

            t_idx, j_idx = np.where(ok)
            keep = (t_idx >= 1)          # defensive
            t_idx = t_idx[keep]
            j_idx = j_idx[keep]

            if t_idx.size == 0:
                continue

            predictors = observations[t_idx - 1, j_idx, :]   # (N,D)
            outcomes   = observations[t_idx,     j_idx, :]   # (N,D)

            lr = LinearRegression(fit_intercept=True)
            lr.fit(predictors, outcomes)

            A = lr.coef_
            b = lr.intercept_

            expectations = (A @ predictors.T).T + b
            residuals = outcomes - expectations

            Q = np.cov(residuals, rowvar=False) + 1e-4 * np.eye(D)

            # broadcast to all entities
            As[:, k] = A
            bs[:, k] = b
            Qs[:, k] = Q
            
    As, bs, Qs = clamp_parameters(observations, As, bs, Qs, K_SPECIAL=0) #Clamp the parameters for the silent state k = 0. In this state, I want this emission.  
    As, bs, Qs = clamp_parameters(observations, As, bs, Qs, K_SPECIAL=1) #Clamp the parameters for the silent state k = 1. In this state, I want this emission.  

    As = jnp.asarray(As)
    bs = jnp.asarray(bs)
    Qs = jnp.asarray(Qs)
    return ContinuousStateParameters_JAX(As, bs, Qs), kms


def make_label_cluster_preinit_CSP_shared_across_entities_JAX(
    DIMS: Dims,
    observations: JaxNumpyArray3D,                  # (T,J,D)
    evid_onehot: np.ndarray,                        # (T,J,C)
    example_end_times: np.ndarray,                  # (N+1,)
    mask_observations: Optional[np.ndarray] = None, # (T,J) bool
    *,
    k_no_evidence: int = 2,
    k_evidence: int = 3,
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> ContinuousStateParameters_JAX:

    if verbose:
        print("Now performing label-based CSP pre-initialization (shared across entities, no k-means).")

    if save_dir is None:
        save_dir = "./"
    os.makedirs(save_dir, exist_ok=True)

    diag_csv   = os.path.join(save_dir, "gaussian_initialization_diagnostics.csv")
    cross_csv  = os.path.join(save_dir, "gaussian_initialization_crosslik.csv")  # NEW
    params_csv = os.path.join(save_dir, "gaussian_initialization_params.csv")

    obs = np.asarray(observations)
    evid = np.asarray(evid_onehot)
    T, J, D = obs.shape
    K = DIMS.K

    As = np.zeros((J, K, D, D))
    bs = np.zeros((J, K, D))
    Qs = np.tile(np.eye(D)[None, None, :, :], (J, K, 1, 1))

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "unsupervised_inference" 
    z = np.load(data_dir / "silence_embedding.npz")
    silence = z["silence"]

    silence_vec = np.asarray(silence)  # (D,)

        # -----------------------------
    # Clamp fixed states
    # -----------------------------
    As, bs, Qs = clamp_parameters(obs, As, bs, Qs, K_SPECIAL=0)
    As, bs, Qs = clamp_parameters(obs, As, bs, Qs, K_SPECIAL=1)
    As = np.array(As, copy=True)
    bs = np.array(bs, copy=True)
    Qs = np.array(Qs, copy=True)
    # Grab clamped silent params (use j0; they're broadcast across entities by clamp)
    j0 = 0
    A0 = np.asarray(As[j0, 0]); b0 = np.asarray(bs[j0, 0]); Q0 = np.asarray(Qs[j0, 0])
    A1 = np.asarray(As[j0, 1]); b1 = np.asarray(bs[j0, 1]); Q1 = np.asarray(Qs[j0, 1])


    def mean_loglik_gaussian_AR(predictors, outcomes, A, b, Q) -> float:
        predictors = np.asarray(predictors)
        outcomes = np.asarray(outcomes)
        A = np.asarray(A)
        b = np.asarray(b)
        Q = np.asarray(Q)

        N, D_ = outcomes.shape
        yhat = (A @ predictors.T).T + b
        resid = outcomes - yhat

        Q_sym = 0.5 * (Q + Q.T)
        L = np.linalg.cholesky(Q_sym)

        z = np.linalg.solve(L, resid.T).T
        maha = np.sum(z**2, axis=1)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))

        loglik = -0.5 * (maha + logdet + D_ * np.log(2.0 * np.pi))
        return float(np.mean(loglik))

    def extract_pooled_ar_pairs_to_silence(
        *,
        observations: np.ndarray,             # (T,J,D)
        example_end_times: np.ndarray,        # (N+1,)
        mask_observations: np.ndarray | None, # (T,J)
        silence_vec: np.ndarray,              # (D,)
        from_non_silent_only: bool = True,    # True = non-silent->silent
    ) -> tuple[np.ndarray, np.ndarray]:

        obs_ = np.asarray(observations)
        T_, J_, D_ = obs_.shape
        s = np.asarray(silence_vec).reshape(1, 1, D_)

        is_silent = np.all(obs_ == s, axis=-1)  # (T_,J_)

        starts = set(int(example_end_times[i - 1] + 1) for i in range(1, len(example_end_times)))
        valid_t = np.ones(T_, dtype=bool)
        for st in starts:
            if 0 <= st < T_:
                valid_t[st] = False

        t_idx = np.arange(1, T_)
        ok_t = valid_t[t_idx]  # (T_-1,)

        outcome_is_silence = is_silent[1:, :]  # (T_-1,J_)
        if from_non_silent_only:
            predictor_ok = ~is_silent[:-1, :]  # (T_-1,J_)
        else:
            predictor_ok = np.ones((T_ - 1, J_), dtype=bool)

        keep = outcome_is_silence & predictor_ok
        keep &= ok_t[:, None]

        if mask_observations is not None:
            m = np.asarray(mask_observations).astype(bool)
            keep &= m[1:, :] & m[:-1, :]

        if not np.any(keep):
            return np.zeros((0, D_)), np.zeros((0, D_))

        predictors = obs_[:-1, :, :][keep]  # (N_pairs, D_)
        outcomes   = obs_[1:,  :, :][keep]  # (N_pairs, D_)
        return predictors, outcomes

    def extract_pooled_ar_pairs_to_silence_by_predictor_evidence(
        *,
        observations: np.ndarray,              # (T,J,D)
        evid_onehot: np.ndarray,               # (T,J,C)
        example_end_times: np.ndarray,         # (N+1,)
        mask_observations: np.ndarray | None,  # (T,J)
        silence_vec: np.ndarray,               # (D,)
        want_evidence: bool,                   # False => predictor is no-evidence, True => predictor has evidence
    ) -> tuple[np.ndarray, np.ndarray]:

        obs_ = np.asarray(observations)
        evid_ = np.asarray(evid_onehot)
        T_, J_, D_ = obs_.shape
        _, _, C_ = evid_.shape

        s = np.asarray(silence_vec).reshape(1, 1, D_)
        is_silent = np.all(obs_ == s, axis=-1)  # (T_,J_)

        # prevent cross-example transitions at the first timestep of each example
        starts = set(int(example_end_times[i - 1] + 1) for i in range(1, len(example_end_times)))
        valid_t = np.ones(T_, dtype=bool)
        for st in starts:
            if 0 <= st < T_:
                valid_t[st] = False

        t_idx = np.arange(1, T_)
        ok_t = valid_t[t_idx]  # (T_-1,)

        # outcome must be silence
        outcome_is_silence = is_silent[1:, :]  # (T_-1,J_)

        # predictor evidence class is determined at time t-1 (i.e., evid_[:-1])
        if C_ <= 1:
            predictor_has_evidence = np.zeros((T_ - 1, J_), dtype=bool)
        else:
            predictor_has_evidence = np.any(evid_[:-1, :, 1:], axis=-1)  # (T_-1,J_)

        predictor_is_no_evidence = ~predictor_has_evidence

        if want_evidence:
            predictor_ok = predictor_has_evidence
        else:
            predictor_ok = predictor_is_no_evidence

        # IMPORTANT: these are talk->silence transitions, not silence->silence
        predictor_ok &= ~is_silent[:-1, :]

        keep = outcome_is_silence & predictor_ok
        keep &= ok_t[:, None]

        if mask_observations is not None:
            m = np.asarray(mask_observations).astype(bool)
            keep &= m[1:, :] & m[:-1, :]

        if not np.any(keep):
            return np.zeros((0, D_)), np.zeros((0, D_))

        predictors = obs_[:-1, :, :][keep]  # (N_pairs, D_)
        outcomes   = obs_[1:,  :, :][keep]  # (N_pairs, D_)
        return predictors, outcomes

    # Precompute to-silence transitions once
    pred_to_sil, out_to_sil = extract_pooled_ar_pairs_to_silence(
        observations=obs,
        example_end_times=example_end_times,
        mask_observations=mask_observations,
        silence_vec=silence_vec,
        from_non_silent_only=True,
    )

    def extract_pooled_ar_pairs_silence_to_silence(
        *,
        observations: np.ndarray,             # (T,J,D)
        example_end_times: np.ndarray,        # (N+1,)
        mask_observations: np.ndarray | None, # (T,J)
        silence_vec: np.ndarray,              # (D,)
    ) -> tuple[np.ndarray, np.ndarray]:

        obs_ = np.asarray(observations)
        T_, J_, D_ = obs_.shape
        s = np.asarray(silence_vec).reshape(1, 1, D_)

        is_silent = np.all(obs_ == s, axis=-1)  # (T_,J_)

        starts = set(int(example_end_times[i - 1] + 1) for i in range(1, len(example_end_times)))
        valid_t = np.ones(T_, dtype=bool)
        for st in starts:
            if 0 <= st < T_:
                valid_t[st] = False

        t_idx = np.arange(1, T_)
        ok_t = valid_t[t_idx]  # (T_-1,)

        keep = (is_silent[1:, :] & is_silent[:-1, :])   # silence -> silence
        keep &= ok_t[:, None]

        if mask_observations is not None:
            m = np.asarray(mask_observations).astype(bool)
            keep &= m[1:, :] & m[:-1, :]

        if not np.any(keep):
            return np.zeros((0, D_)), np.zeros((0, D_))

        predictors = obs_[:-1, :, :][keep]
        outcomes   = obs_[1:,  :, :][keep]
        return predictors, outcomes


    N_to_sil = int(out_to_sil.shape[0])

    # --- Build evaluation datasets for "silent-state likelihoods" ---
    pred_sil_sil, out_sil_sil = extract_pooled_ar_pairs_silence_to_silence(
        observations=obs,
        example_end_times=example_end_times,
        mask_observations=mask_observations,
        silence_vec=silence_vec,
    )

    pred_noev_to_sil, out_noev_to_sil = extract_pooled_ar_pairs_to_silence_by_predictor_evidence(
        observations=obs,
        evid_onehot=evid,
        example_end_times=example_end_times,
        mask_observations=mask_observations,
        silence_vec=silence_vec,
        want_evidence=False,   # predictor is talk with NO evidence
    )

    pred_ev_to_sil, out_ev_to_sil = extract_pooled_ar_pairs_to_silence_by_predictor_evidence(
        observations=obs,
        evid_onehot=evid,
        example_end_times=example_end_times,
        mask_observations=mask_observations,
        silence_vec=silence_vec,
        want_evidence=True,    # predictor is talk WITH evidence
    )

    # k=0 eval set = (sil->sil) union (no-evidence talk->sil)
    pred_k0 = np.vstack([pred_sil_sil, pred_noev_to_sil]) if (pred_sil_sil.shape[0] + pred_noev_to_sil.shape[0]) > 0 else np.zeros((0, D))
    out_k0  = np.vstack([out_sil_sil,  out_noev_to_sil])  if (out_sil_sil.shape[0]  + out_noev_to_sil.shape[0])  > 0 else np.zeros((0, D))

    # k=1 eval set = (evidence talk->sil)
    pred_k1 = pred_ev_to_sil
    out_k1  = out_ev_to_sil

    N_k0 = int(out_k0.shape[0])
    N_k1 = int(out_k1.shape[0])

    ll_k0 = float("nan") if N_k0 == 0 else mean_loglik_gaussian_AR(pred_k0, out_k0, A0, b0, Q0)
    ll_k1 = float("nan") if N_k1 == 0 else mean_loglik_gaussian_AR(pred_k1, out_k1, A1, b1, Q1)

    # Save once
    silent_csv = os.path.join(save_dir, "gaussian_initialization_silentlik.csv")
    pd.DataFrame([{
        "N_sil_to_sil": int(out_sil_sil.shape[0]),
        "N_noev_to_sil": int(out_noev_to_sil.shape[0]),
        "N_ev_to_sil": int(out_ev_to_sil.shape[0]),
        "N_eval_k0_total": int(N_k0),
        "N_eval_k1_total": int(N_k1),
        "mean_loglik_eval_k0_under_clamped_k0": float(ll_k0),
        "mean_loglik_eval_k1_under_clamped_k1": float(ll_k1),
    }]).to_csv(silent_csv, index=False)


    # -----------------------------
    # Fit + broadcast + log diagnostics
    # -----------------------------
    def fit_and_broadcast_for_k(k: int, want_evidence: bool):
        predictors, outcomes, _weights = extract_pooled_ar_pairs_by_evidence_with_silent_predictor(
            observations=obs,
            evid_onehot=evid,
            example_end_times=example_end_times,
            mask_observations=mask_observations,
            want_evidence=want_evidence,
        )

        if outcomes.shape[0] == 0:
            if verbose:
                print(f"[warn] No silent->non-silent pairs found for k={k} (want_evidence={want_evidence}). "
                      f"Leaving defaults for this k.")
            return None

        X = np.hstack([predictors, np.ones((predictors.shape[0], 1))])

        lr = Ridge(alpha=1.0, fit_intercept=False)
        lr.fit(X, outcomes)

        A = lr.coef_[:, :-1]   # (D,D)
        b = lr.coef_[:, -1]    # (D,)

        yhat = (A @ predictors.T).T + b
        resid = outcomes - yhat

        Q = np.cov(resid, rowvar=False)
        if Q.ndim == 0:
            Q = np.array([[float(Q)]])
        Q = Q + 1e-4 * np.eye(D)

        # broadcast
        As[:, k] = A
        bs[:, k] = b
        Qs[:, k] = Q

        # self mean loglik
        N_pairs, D_ = outcomes.shape
        mean_loglik = mean_loglik_gaussian_AR(predictors, outcomes, A, b, Q)

        mean_loglik_under_k0_silent = mean_loglik_gaussian_AR(predictors, outcomes, A0, b0, Q0)

        # to-silence mean loglik under these params
        if N_to_sil == 0:
            mean_loglik_to_silence = float("nan")
        else:
            mean_loglik_to_silence = mean_loglik_gaussian_AR(pred_to_sil, out_to_sil, A, b, Q)


        row = pd.DataFrame([{
            "k": int(k),
            "want_evidence": bool(want_evidence),
            "N_pairs": int(N_pairs),
            "D": int(D_),
            "mean_loglik": float(mean_loglik),
            "mean_loglik_under_k0_silent": float(mean_loglik_under_k0_silent),
            "N_to_silence": int(N_to_sil),
            "mean_loglik_to_silence": float(mean_loglik_to_silence),
        }])
        row.to_csv(diag_csv, mode="a", header=not Path(diag_csv).exists(), index=False)

        # NEW: return what we need for cross-evals
        return {
            "k": int(k),
            "want_evidence": bool(want_evidence),
            "predictors": predictors,
            "outcomes": outcomes,
            "A": A,
            "b": b,
            "Q": Q,
            "N_pairs": int(N_pairs),
            "D": int(D_),
        }

    # -----------------------------
    # Fit both regimes
    # -----------------------------
    fit_no = fit_and_broadcast_for_k(k_no_evidence, want_evidence=False)
    fit_ev = fit_and_broadcast_for_k(k_evidence, want_evidence=True)

    # -----------------------------
    # NEW: Cross mean log-likelihoods (k2 data under k3 params, and vice versa)
    # -----------------------------
    if (fit_no is not None) and (fit_ev is not None):
        # data(no) under params(ev)
        ll_no_under_ev = mean_loglik_gaussian_AR(
            fit_no["predictors"], fit_no["outcomes"],
            fit_ev["A"], fit_ev["b"], fit_ev["Q"]
        )
        # data(ev) under params(no)
        ll_ev_under_no = mean_loglik_gaussian_AR(
            fit_ev["predictors"], fit_ev["outcomes"],
            fit_no["A"], fit_no["b"], fit_no["Q"]
        )

        # (optional but useful) to-silence under each other's params too
        if N_to_sil == 0:
            ll_to_sil_under_no = float("nan")
            ll_to_sil_under_ev = float("nan")
        else:
            ll_to_sil_under_no = mean_loglik_gaussian_AR(pred_to_sil, out_to_sil, fit_no["A"], fit_no["b"], fit_no["Q"])
            ll_to_sil_under_ev = mean_loglik_gaussian_AR(pred_to_sil, out_to_sil, fit_ev["A"], fit_ev["b"], fit_ev["Q"])

        cross_rows = pd.DataFrame([
            {
                "data_k": fit_no["k"],
                "data_want_evidence": fit_no["want_evidence"],
                "param_k": fit_ev["k"],
                "param_want_evidence": fit_ev["want_evidence"],
                "N_pairs_data": fit_no["N_pairs"],
                "D": fit_no["D"],
                "mean_loglik_cross": float(ll_no_under_ev),
                "N_to_silence": int(N_to_sil),
                "mean_loglik_to_silence_under_params": float(ll_to_sil_under_ev),
            },
            {
                "data_k": fit_ev["k"],
                "data_want_evidence": fit_ev["want_evidence"],
                "param_k": fit_no["k"],
                "param_want_evidence": fit_no["want_evidence"],
                "N_pairs_data": fit_ev["N_pairs"],
                "D": fit_ev["D"],
                "mean_loglik_cross": float(ll_ev_under_no),
                "N_to_silence": int(N_to_sil),
                "mean_loglik_to_silence_under_params": float(ll_to_sil_under_no),
            },
        ])

        cross_rows.to_csv(cross_csv, mode="a", header=not Path(cross_csv).exists(), index=False)

    elif verbose:
        print("[warn] Skipping cross-likelihoods because one of the fits had no data.")


    # -----------------------------
    # Save params for inspection
    # -----------------------------
    j0 = 0
    rows = []
    for k in range(K):
        A = np.asarray(As[j0, k])
        b = np.asarray(bs[j0, k])
        Q = np.asarray(Qs[j0, k])
        rows.append({
            "k": int(k),
            "A": json.dumps(A.tolist()),
            "b": json.dumps(b.tolist()),
            "Q": json.dumps(Q.tolist()),
        })
    pd.DataFrame(rows).to_csv(params_csv, index=False)

    return ContinuousStateParameters_JAX(jnp.asarray(As), jnp.asarray(bs), jnp.asarray(Qs))

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
            outside_entity_recurrence,
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
    seed: int = 126,
    mask_observations: Optional[JaxNumpyArray2D] = None,
    save_dir: Optional[str] = None,
    treat_ETP_params_as_tpm_during_bottom_half_inference: bool = False,
    params_frozen: Optional[AllParameters_JAX] = None,
    outside_system_recurrence: Optional[JaxNumpyArray2D] = None,
    outside_entity_recurrence: Optional[JaxNumpyArray3D] = None,
    verbose: bool = True,
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
        params_frozen: frozen version of all params STP, ETP, CSP, IP. 
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

    if params_frozen: # If params frozen is provided, use these params. If not, create the params. 
        CSP_JAX = params_frozen.CSP
        ETP_JAX = params_frozen.ETP
        IP_JAX = params_frozen.IP

    else:
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data" / "unsupervised_inference" / "initialization"
        init_data = np.load(data_dir / "initialization_dataset.npz", allow_pickle=True)
        all_X = init_data["X"].tolist()     
        all_Y = init_data["Y"].tolist()
        init_observations = np.concatenate(all_X, axis=0)
        init_example_end_times = init_data["example_end_times"].tolist()
        init_evid_onehot = np.concatenate(all_Y, axis=0)

        CSP_JAX = make_label_cluster_preinit_CSP_shared_across_entities_JAX(
            DIMS,
            init_observations,
            init_evid_onehot, 
            init_example_end_times,
            save_dir = save_dir,
        )

        ETP_JAX = make_data_free_preinitialization_of_ETP_JAX(
            DIMS, method_for_Psis="hard_code", seed=seed, fixed_self_transition_prob=0.25,)
        IP_JAX = make_data_free_preinitialization_of_IP_JAX(DIMS, observations)


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
            method_for_Upsilon="hard_code",
            fixed_self_transition_prob=0.5,
            seed=seed,
        )
  

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
