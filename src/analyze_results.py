from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]

test_results_dir = repo_root / "results" / "unsupervised_inference_test"

dataset = "new" # "seen" or "new"
agg = 126 # "mean" or "median" or seed no.

MODEL_ADJUSTMENTS = ["perf_ev", "class_ev", "class_ev_no_sys", "no_ev", "kmeans_init_full_ev", "kmeans_init_noisy_ev"]

def create_results_df(test_results_dir, dataset, agg, artifact_name, group_by=None):
    """
    Purpose:
        Creates a DataFrame containing aggregated performance across seeds for each ablation.

    Args:
        test_results_dir (Path): Directory containing test results
        dataset (str): Dataset name ("seen" or "new")
        agg (int or str): Aggregation method ("mean", "median") or specific seed number
        artifact_name (str): Name of the artifact to analyze
        group_by (list or None): Columns to group by for aggregation. Defaults to None.
    """
    results_df = pd.DataFrame()
    for adjustment in MODEL_ADJUSTMENTS:
        df = pd.DataFrame()
        for seed in [17, 76, 126]:
            run_dir = test_results_dir / f"dataset_{dataset}_problem" / f"adjustment_{adjustment}" / f"seed_{seed}"
            df_seed = pd.read_csv(run_dir / "artifacts" / f"{artifact_name}.csv")
            df_seed["seed"] = seed
            df = pd.concat([df, df_seed], ignore_index=True)
        
        if agg == "mean":
            df_agg = df.groupby(by=group_by).mean().reset_index()
            df_agg["seed"] = "mean"
        elif agg == "median":
            df_agg = df.groupby(by=group_by).median().reset_index()
            df_agg["seed"] = "median"
        elif agg == None:
            df_agg = df
        else:
            df_agg = df[df["seed"] == int(agg)]
        
        df_agg.insert(0, "seed", df_agg.pop("seed"))
        df_agg.insert(0, "adjustment", adjustment)
        results_df = pd.concat([results_df, df_agg], ignore_index=True)
        
        
    results_df.to_csv(repo_root / "results" / "unsupervised_inference_test" / f"dataset_{dataset}_problem" / f"{artifact_name}_{agg}.csv", index=False)

group_by = {
    "entity_correlation_table": "entity",
    "speaker_transition_to_silence_k1_by_mech_evidence": ["scenario", "k_target"],
    "transition_posteriors_conditioned_on_speaker_evidence": ["condition", "transition", "k_used"]
}

# for artifact in ["entity_correlation_table", "speaker_transition_to_silence_k1_by_mech_evidence", "transition_posteriors_conditioned_on_speaker_evidence"]:
#     create_results_df(test_results_dir, dataset, agg, artifact, group_by.get(artifact, None))

train_runs = [
    "seed_{seed}_system_size_2_n_iterations_15_adjustment_None_full_evidence",
    "seed_{seed}_system_size_2_n_iterations_15_adjustment_None_noisy_evidence",
    "seed_{seed}_system_size_1_n_iterations_15_adjustment_one_system_regime_noisy_evidence",
    "seed_{seed}_system_size_2_n_iterations_15_adjustment_remove_recurrence_no_evidence",
    "seed_{seed}_system_size_2_n_iterations_15_adjustment_kmeans_full_evidence",
    "seed_{seed}_system_size_2_n_iterations_15_adjustment_kmeans_noisy_evidence",
]

# Combine results across training seeds for each artifact into a single DataFrame
for artifact in ["entity_correlation_table", "speaker_transition_to_silence_k1_by_mech_evidence", "transition_posteriors_conditioned_on_speaker_evidence"]:
    for run in train_runs:
        full_df = pd.DataFrame()
        for seed in [17, 41, 76, 166, 126]:
            run_dir = repo_root / "results" / "unsupervised_inference" / run.format(seed=seed) / "artifacts"
            df = pd.read_csv(run_dir / f"{artifact}.csv")
            df.insert(0, "seed", seed)
            full_df = pd.concat([full_df, df], ignore_index=True)
        
        full_df.to_csv(repo_root / "results" / "unsupervised_inference" / f"{run.split('adjustment_')[1].split('_evidence')[0]}_{artifact}_all_seeds.csv", index=False)