import pandas as pd
from pathlib import Path
import numpy as np

def highlight_max_row(df: pd.DataFrame, column: str) -> list:
    """
    Purpose: Prints the DataFrame to terminal with the row containing
    the maximum value in `column` highlighted green.
    
    Arguments:
        df: DataFrame to print
        column: str of the column to find the max value from and highlight
    
    Returns:
        List of seed(s) corresponding to the row(s) with the max value in `column`
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    max_value = df[column].max()

    GREEN = "\033[92m"
    RESET = "\033[0m"

    print("  ".join(str(c) for c in df.columns))

    for _, row in df.iterrows():
        row_str = "  ".join(str(v) for v in row.values)

        if row[column] == max_value:
            print(f"{GREEN}{row_str}{RESET}")
        else:
            print(row_str)

    return df.loc[df[column] == max_value, 'seed'].values.tolist()

repo_root = Path(__file__).resolve().parents[1]
train_results_dir = repo_root / "results" / "unsupervised_inference"

run_file_names = [
    "None_full_{artifact}_all_seeds.csv",
    "None_noisy_{artifact}_all_seeds.csv",
    "one_system_regime_noisy_{artifact}_all_seeds.csv",
    "remove_recurrence_no_{artifact}_all_seeds.csv",
    "kmeans_full_{artifact}_all_seeds.csv",
    "kmeans_noisy_{artifact}_all_seeds.csv",
]

artifacts = ["entity_correlation_table", "speaker_transition_to_silence_k1_by_mech_evidence", "transition_posteriors_conditioned_on_speaker_evidence"]

best_seed_counts = {}

for filename in run_file_names:
    for artifact in artifacts:
        df = pd.read_csv(train_results_dir / filename.format(artifact=artifact))
        
        print(f"Filename: {filename.format(artifact=artifact)}")
        
        # Performance measured by highest mean pearson correlation across entities
        if artifact == "entity_correlation_table":
            # only keep pearson correlation columns
            pearson_cols = [c for c in df.columns if c.startswith("pearson")]

            # mean per seed per pearson correlation column
            df_agg = (
                df.groupby("seed")[pearson_cols]
                .mean()
                .reset_index()
            )

            # for each seed, get the pearson column with the highest mean
            df_agg["best_pearson_value"] = df_agg[pearson_cols].max(axis=1)
            
            df_agg = df_agg[["seed", "best_pearson_value"]]

            best_seeds = highlight_max_row(df_agg, "best_pearson_value")
        
        # Performance measured by largest difference in mean posterior for k=1 between evidence and no evidence conditions
        if artifact == "speaker_transition_to_silence_k1_by_mech_evidence":
            df_agg = df.groupby(by=["seed"])["mean_posterior_k"].agg(lambda x: x.max() - x.min()).reset_index()
            best_seeds = highlight_max_row(df_agg, "mean_posterior_k")
        
        # Performance measured by largest average difference when moving from no evidence to evidence across both silent-to-talk and silent-to-silent transitions
        if artifact == "transition_posteriors_conditioned_on_speaker_evidence":
            df_agg = df.groupby('seed').apply(func=
                lambda g: (g.loc[(g['condition']=='evidence') & (g['transition']=='silent_to_talk'), 'mean_posterior'].values[0] 
                        - g.loc[(g['condition']=='no_evidence') & (g['transition']=='silent_to_talk'), 'mean_posterior'].values[0] +
                        g.loc[(g['condition']=='evidence') & (g['transition']=='silent_to_silent'), 'mean_posterior'].values[0] 
                        - g.loc[(g['condition']=='no_evidence') & (g['transition']=='silent_to_silent'), 'mean_posterior'].values[0]
                ) / 2, include_groups=False).reset_index(name='mean_posterior_diff')
            best_seeds = highlight_max_row(df_agg, "mean_posterior_diff")
        
        for best_seed in best_seeds:
            best_seed_counts[best_seed] = best_seed_counts.get(best_seed, 0) + 1
        
        print('\n\n')

print("Best seed counts across all runs:")
for seed, count in best_seed_counts.items():
    print(f"Seed {seed}: {count} times")