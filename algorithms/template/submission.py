"""Template submission module.

See https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#allowed-submissions
and https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#disallowed-submissions
for guidelines.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from algoperf import spec


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Creates a Nesterov optimizer and a learning rate schedule.
  Returns: spec.OptimizerState initialized optimizer state
  """
  pass


def update_params(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  batch: Dict[str, spec.Tensor],
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
  train_state: Optional[Dict[str, Any]] = None,
) -> spec.UpdateReturn:
  """
  Returns:
   spec.OptimizerState: new optimizer state
   spec.ParameterTypeTree: new params
   new_model_state: new model state
  """
  pass


def prepare_for_eval(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
) -> spec.UpdateReturn:
  """
  Returns:
   new_optimizer_state
   new_params
   new_model_state
  """
  pass


def get_batch_size(workload_name):
  """
  Gets batch size for workload.
  Note that these batch sizes only apply during training and not during evals.
  Args:
    workload_name (str): Valid workload_name values are: "wmt", "ogbg",
      "criteo1tb", "fastmri", "imagenet_resnet", "imagenet_vit",
      "librispeech_deepspeech", "librispeech_conformer" or any of the
      variants.
  Returns:
    int: batch_size
  Raises:
    ValueError: If workload_name is not handled.
  """
  pass


def data_selection(
  workload: spec.Workload,
  input_queue: Iterator[Dict[str, spec.Tensor]],
  optimizer_state: spec.OptimizerState,
  current_param_container: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  global_step: int,
  rng: spec.RandomState,
) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  Tip:
  If you would just like the next batch from the input queue return next(input_queue).

  Returns:
   batch: next batch of input data
  """
  pass
