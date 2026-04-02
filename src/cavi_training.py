import warnings
from typing import Optional, Tuple, Union
import numpy as np

from utilities.util import example_end_times_are_proper, append_gaussian_params_by_iter_csv
from compute_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from utilities.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
)
from model import Model 
from prior import SystemTransitionPrior_JAX
from params import AllParameters_JAX, dims_from_params
import compute_ELBO as elbo_utils
from metrics import get_entity_correlation_metric, get_system_correlation_metric


from expectation_step import run_VES_step_JAX, run_VEZ_step_JAX
from maximization_step import (
    M_Step_Toggle_Value,
    M_Step_Toggles,
    run_M_step_for_CSP,
    run_M_step_for_ETP,
    run_M_step_for_IP,
    run_M_step_for_STP,
)

"""
Uses a Coordinate Ascent Variational Inference (CAVI) method to train the HSRDM. Runs both the E-step computed in -> (expectation_step.py)
and the M-step computed in -> (maximizations-step.py) for a user specified number of iterations. 
"""

def run_CAVI_with_JAX(
    all_params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX] = None,
    model: Model = None,
    observations: Union[JaxNumpyArray2D, JaxNumpyArray3D] = None, 
    example_end_times: Optional[JaxNumpyArray1D] = None,
    n_iterations: int = 1,
    M_step_toggles: Optional[M_Step_Toggles] = None,
    num_M_step_iters: int = 50,
    outside_system_recurrence: Optional[JaxNumpyArray2D] = None,
    outside_entity_recurrence: Optional[JaxNumpyArray3D] = None,
    mask_observations: Optional[NumpyArray2D] = None,
    evid_onehot: Optional[NumpyArray3D] = None,
    verbose: bool = True,
    save_dir: Optional[str] = None, 
) -> Tuple[HMM_Posterior_Summary_JAX, HMM_Posterior_Summaries_JAX, AllParameters_JAX]:
    """

    Purpose: Implemnt the variational inference procedure for the MODEL parameters and the assumed posterior distribution.
    The goal is to find the posterior that maximizes the Evidence Lower Bound (ELBO). The E-step finds the current optimal 
    posterior given fixed MODEL params and the M-step finds improved MODEL params given the current optimal posterior. The 
    ELBO is maximized over iterations. 

    Arguments:
        all_params: STP, ETP, CSP, IP
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions.
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        system_transition_prior: Dirichlet distribution over the categorical parameters in -> (prior.py)
        model: joint distribution defined in -> (model.py)
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        n_iterations: total number of CAVI iterations
        M_step_toggles: Toggle value for the optimization setting (e.g. closed_form, gradient_descent)
        num_M_step_iters: number of iterations for optimization (e.g. gradient descent)
        outside_system_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 
        outside_entity_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        evid_onehot: One hot vector labels that are the annotated class evidence of mechanistic reasoning
        verbose: True boolean if we want to print the correlation during training of the system and entity expected regimes
        at the next time step and the current evidence strength of the observation 
        save_dir: str for the path to save the file
    Returns:
        VES_Summary, VEZ_Summaries, all parameters.

    """

    ###
    # SET-UP
    ###
    
    DIMS = dims_from_params(all_params)
    T = np.shape(observations)[0]
    ed_list = list()

    if observations.ndim == 2:
        print("Continuous states has only two array dimensions; now adding a third array dimension with 1 element.")
        observations = observations[:, :, None]

    if mask_observations is None:
        mask_observations = np.full((T, DIMS.J), True)
    else:
        # TODO: Raise error if mask_observations has False followed by True for any entity j;
        # in the current implementation, inference will not be done correctly, because the VEZ step will not correctly
        # remove the missing data -- so the inference after the missing data will be artificially good.
        warnings.warn(
            f"Selecting only some observations for usage correctly alters inference -- the M-step and VES "
            f"steps are changed so as to remove the influence of unused states.  However, the "
            f"ELBO is still computed on the full dataset. This is because of the current implementation: under "
            f"the hood, we compute the full VEZ step, and only ablate post-hoc."
        )

    if False in mask_observations[0]:
        raise NotImplementedError(
            f"We currently assume the initial observations is used for all entities. "
            f"The implementation can be changed to handle this case, though: Update the "
            f"code for doing the M-step for the initialization parameters."
        )

    if example_end_times is None:
        example_end_times = np.array([-1, T])

    if not example_end_times_are_proper(example_end_times, len(observations)):
        raise ValueError(
            f"Event end times do not have the proper format. Consult the `examples` module "
            f"and try again.  `example_end_times` MUST begin with -1 and end with T, the length "
            f"of the grand time series."
        )

    if DIMS.L == 1:
        # Automatically turn off M-step for system level parameters if there is only one system state.
        M_step_toggles.STP = M_Step_Toggle_Value.OFF


    def local_calc_elbo(**kws):
        all_params = kws.get('all_params')
        VES_summary, VEZ_summaries = kws.get('VES_summary'), kws.get('VEZ_summaries')
        STP_prior = kws.get('system_transition_prior')
        model = kws.get('model')
        data_TJD, example_end_times, mask_TJ = [
            kws.get(s) for s in [
                'observations', 'example_end_times', 'mask_observations']]
        elbo_dict = elbo_utils.calc_elbo(
            all_params, VES_summary, VEZ_summaries, STP_prior,
            model, data_TJD, example_end_times, mask_TJ, outside_system_recurrence, outside_entity_recurrence, return_dict=True)
        return elbo_dict
    def pretty_print_elbo(**elbo_kws):
        fstr = "elbo={elbo:9.5f} energy={energy:9.5f} entrp={entropy:9.5f}"
        msg = fstr.format(**elbo_kws)
        try:
            print("%-32s" % elbo_kws["status"], msg)
        except KeyError:
            print(msg)


    elbo_dict = local_calc_elbo(**locals())
    elbo_dict['status'] = "at init"
    ed_list.append(elbo_dict)
    if verbose:
        pretty_print_elbo(**elbo_dict)

    ###
    # CAVI
    ###

    for i in range(n_iterations):
        print(f"\n ---- Now running iteration {i+1} ----")

        VES_summary = run_VES_step_JAX(
            all_params.STP,
            all_params.ETP,
            all_params.IP,
            observations,
            VEZ_summaries,
            model,
            example_end_times,
            outside_system_recurrence,
            outside_entity_recurrence,
            mask_observations,
        )
        elbo_dict = local_calc_elbo(**locals())
        elbo_dict['status'] = f"iter {i:3d} after VES"
        ed_list.append(elbo_dict)
        if verbose:
            pretty_print_elbo(**elbo_dict)

        VEZ_summaries = run_VEZ_step_JAX(
            all_params.CSP,
            all_params.ETP,
            all_params.IP,
            observations,
            VES_summary.expected_regimes,
            model,
            example_end_times,
            outside_entity_recurrence,
            one_hot_evid = evid_onehot,
            iteration = i, 
            save_dir = save_dir,
            make_table = True, 
        )
        elbo_dict = local_calc_elbo(**locals())
        elbo_dict['status'] = f"iter {i:3d} after VEZ"
        ed_list.append(elbo_dict)
        if verbose:
            pretty_print_elbo(**elbo_dict)
     
        ###
        # M-step (ETP)
        ###

        all_params = run_M_step_for_ETP(
            all_params,
            M_step_toggles.ETP,
            VES_summary,
            VEZ_summaries,
            observations,
            i,
            num_M_step_iters,
            model,
            example_end_times,
            outside_entity_recurrence,
            mask_observations,
            verbose,
        )
        elbo_dict = local_calc_elbo(**locals())
        elbo_dict['status'] = f"iter {i:3d} after Mstep:ETP"
        ed_list.append(elbo_dict)
        if verbose:
            pretty_print_elbo(**elbo_dict)


        ###
        # M-step (STP)
        ###

        all_params = run_M_step_for_STP(
            all_params,
            M_step_toggles.STP,
            VES_summary,
            system_transition_prior,
            i,
            num_M_step_iters,
            model,
            example_end_times,
            outside_system_recurrence,
            observations,
            verbose-1,
        )
        elbo_dict = local_calc_elbo(**locals())
        elbo_dict['status'] = f"iter {i:3d} after Mstep:STP"
        ed_list.append(elbo_dict)
        if verbose:
            pretty_print_elbo(**elbo_dict)


        ###
        # M-step (CSP)
        ###

        all_params = run_M_step_for_CSP(
            all_params,
            M_step_toggles.CSP,
            VEZ_summaries,
            observations,
            i,
            num_M_step_iters,
            model,
            example_end_times,
            mask_observations,
        )

        append_gaussian_params_by_iter_csv(
            save_dir=save_dir,
            iteration=i,
            As=all_params.CSP.As,   
            bs=all_params.CSP.bs,
            Qs=all_params.CSP.Qs,
            j0=0,
        )
        elbo_dict = local_calc_elbo(**locals())
        elbo_dict['status'] = f"iter {i:3d} after Mstep:CSP"
        ed_list.append(elbo_dict)
        if verbose:
            pretty_print_elbo(**elbo_dict)

        ###
        # M-step (IP)
        ###

        IP_new = run_M_step_for_IP(
            all_params.IP,
            M_step_toggles.IP,
            VES_summary,
            VEZ_summaries,
            observations,
            example_end_times,
        )
        all_params = AllParameters_JAX(all_params.STP, all_params.ETP, all_params.CSP, IP_new)
        elbo_dict = local_calc_elbo(**locals())
        elbo_dict['status'] = f"iter {i:3d} after Mstep:IP"
        ed_list.append(elbo_dict)
        if verbose:
            pretty_print_elbo(**elbo_dict)

        
        if verbose:
            entity_correaltions = get_entity_correlation_metric(VEZ_summaries.expected_regimes, evid_onehot)
            system_correaltions = get_system_correlation_metric(VES_summary.expected_regimes, evid_onehot, observations)
            print(f"Entity Correlations: {entity_correaltions}")
            print(f"System Correlations: {system_correaltions}")
        
 
      
    return VES_summary, VEZ_summaries, all_params, ed_list, 
