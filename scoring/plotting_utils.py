import operator
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from performance_profile import check_if_minimized
from scoring_utils import get_experiment_df, get_workload_metrics_and_targets

from algorithmic_efficiency.workloads.workloads import get_base_workload_name

MAX_BUDGETS = {
    'criteo1tb': 7703,
    'fastmri': 8859,
    'imagenet_resnet': 63_008,
    'imagenet_vit': 77_520,
    'librispeech_conformer': 61_068,
    'librispeech_deepspeech': 55_506,
    'ogbg': 18_477,
    'wmt': 48_151,
}

STEP_HINTS = {
    'criteo1tb': 10_666,
    'fastmri': 36_189,
    'imagenet_resnet': 186_666,
    'imagenet_vit': 186_666,
    'librispeech_conformer': 80_000,
    'librispeech_deepspeech': 48_000,
    'ogbg': 80_000,
    'wmt': 133_333,
}

METRIC_FMT = {
    "bleu": "BLEU",
    "loss": "Cross-Entropy Loss",
    "accuracy": "Accuracy",
    "ssim": "SSIM",
    "wer": "Word Error Rate",
    "mean_average_precision": "Mean Average Precision",
}

PLOT_STYLE = {
    "prize_qualification_baseline": {
        "color": "#32414b",
        "linestyle": "--",
        "marker": "None",
        "label": "Baseline",
        "command": r'\baseline',
    },
    "schedule_free_adamw": {
        "color": sns.color_palette()[0],
        "linestyle": "-",
        "marker": "o",
        "label": "Schedule Free AdamW",
        "command": r'\sfadam',
    },
    "AdamG": {
        "color": sns.color_palette()[2],
        "linestyle": "-",
        "marker": "*",
        "label": "AdamG",
        "command": r'\adamg',
    },
    "sinv6_75": {
        "color": sns.color_palette()[4],
        "linestyle": "-",
        "marker": "X",
        "label": "Sinv6 75",
        "command": r'\sinvnum',
    },
    "sinv6": {
        "color": sns.color_palette()[3],
        "linestyle": "-",
        "marker": 7,
        "label": "Sinv6",
        "command": r'\sinv',
    },
    "nadamw_sequential": {
        "color": sns.color_palette()[1],
        "linestyle": "-",
        "marker": "d",
        "label": "NadamW Sequential",
        "command": r'\nadamwseq',
    },
    "shampoo_submission": {
        "color": sns.color_palette("tab10")[9],
        "linestyle": "-",
        "marker": "X",
        "label": "Shampoo",
        "command": r'\shampoosub',
    },
    "caspr_adaptive": {
        "color": sns.color_palette("tab10")[5],
        "linestyle": "-",
        "marker": "*",
        "label": "CASPR Adaptive",
        "command": r'\caspr',
    },
    "schedule_free_prodigy": {
        "color": "dimgray",
        "linestyle": "-",
        "marker": "None",
        "label": "Schedule Free Prodigy",
        "command": r'\sfprodigy',
    },
    "amos": {
        "color": sns.color_palette("tab10")[8],
        "linestyle": "-",
        "marker": "X",
        "label": "Amos",
        "command": r'\amos',
    },
    "lawa_ema": {
        "color": sns.color_palette("pastel")[4],
        "linestyle": "-",
        "marker": 7,
        "label": "Lawa EMA",
        "command": r'\lawaema',
    },
    "lawa_queue": {
        "color": sns.color_palette("tab10")[4],
        "linestyle": "-",
        "marker": "d",
        "label": "Lawa Queue",
        "command": r'\lawaq',
    },
    "cyclic_lr": {
        "color": sns.color_palette("tab10")[2],
        "linestyle": "-",
        "marker": 7,
        "label": "Cyclic LR",
        "command": r'\cycliclr',
    },
    "generalized_adam": {
        "color": sns.color_palette()[1],
        "linestyle": "-",
        "marker": "o",
        "label": "Generalized Adam",
        "command": r'\generalizedadam',
    },
    "nadamp": {
        "color": sns.color_palette("tab10")[3],
        "linestyle": "-",
        "marker": "d",
        "label": "NadamP",
        "command": r'\nadamp',
    },
}


def read_data_from_logs(log_path):
  results = {}

  for team in os.listdir(log_path):
    for submission in os.listdir(os.path.join(log_path, team)):
      print(f"Reading data for submission: {submission}")
      experiment_path = os.path.join(log_path, team, submission)
      df = get_experiment_df(experiment_path)
      results[submission] = df

  return results


def get_time_to_target(trial_df, performance_metric, target):
  # Check if the validation metric is minimized or maximized
  is_minimized = check_if_minimized(performance_metric)
  op = operator.le if is_minimized else operator.ge

  performance = trial_df[performance_metric]
  runtime = trial_df["score"].to_numpy()[0]

  # Check if the target is reached
  target_reached = performance.apply(lambda x: op(x, target))

  # Remove trials that never reach the target
  target_reached = target_reached[target_reached.apply(np.any)]
  # If no trials reach the target return -inf. Else, return the eval index
  # of the earliest point the target is reached.
  if len(target_reached) == 0:
    return np.inf
  else:
    # Find first occurence of True in target_reached
    index_reached = target_reached.apply(np.argmax)
    # Get the runtime at the index
    time_to_target = runtime[index_reached]
    return time_to_target[0]


def clean_submission_results(submission_results, self_tuning=False):
  clean_results = []
  # For each workload check studies
  for workload, group in submission_results.groupby("workload"):
    workload_name = re.sub(r'_(jax|pytorch)$', '', workload)
    validation_metric, validation_target = get_workload_metrics_and_targets(workload)
    base_workload_name = get_base_workload_name(workload_name)
    runtime_budget = MAX_BUDGETS[
        base_workload_name] if not self_tuning else 3 * MAX_BUDGETS[
            base_workload_name]
    step_hint = STEP_HINTS[
        base_workload_name] if not self_tuning else 3 * STEP_HINTS[
            base_workload_name]
    group["last_global_step"] = group["global_step"].apply(lambda x: x[-1])
    group["last_submission_time"] = group["accumulated_submission_time"].apply(
        lambda x: x[-1])
    group["last_performance"] = group[validation_metric].apply(lambda x: x[-1])
    # For each study check trials
    for study, group in group.groupby('study'):
      for trial, group in group.groupby('trial'):
        time_to_target = get_time_to_target(group,
                                            validation_metric,
                                            validation_target)
        clean_results.append({
            'workload': workload_name,
            'study': study,
            'trial': trial[0],
            'time_to_target': time_to_target,
            'runtime': group["score"].values[0],
            'performance': group[validation_metric].values[0],
            'validation_metric': validation_metric,
            'performance_target': validation_target,
            'runtime_budget': runtime_budget,
            'last_global_step': group["last_global_step"].values[0],
            'last_submission_time': group["last_submission_time"].values[0],
            'last_performance': group["last_performance"].values[0],
            'max_global_step': step_hint,
        })
  df = pd.DataFrame.from_records(clean_results)
  return df


def clean_results(results, self_tuning=False):
  clean_results = {}
  for submission_name, submission_results in results.items():
    clean_results[submission_name] = clean_submission_results(
        submission_results, self_tuning)

  return clean_results
