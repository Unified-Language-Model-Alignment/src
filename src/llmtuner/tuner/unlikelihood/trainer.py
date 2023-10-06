# Authors: Tianchi Cai, Xierui Song 2023

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import DataCollator, BatchEncoding
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments

from trl.trainer.utils import disable_dropout_in_model, pad_to_length
from torch.nn import CrossEntropyLoss
import numpy as np

from llmtuner.extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class UnlikelihoodTrainer(Trainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        disable_dropout: Optional[bool] = True,
        sft_loss_type: Optional[Literal["logloss", "shifted_ce_loss"]] = "shifted_ce_loss",
        **kwargs
    ):
        if disable_dropout:
            disable_dropout_in_model(model)

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.use_unlikelihood_data_collator = True  # hack to avoid warning
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.sft_loss_type = sft_loss_type
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids',
                   which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if self.is_encoder_decoder:
            max_length = batch["labels"].shape[1]
        else:
            max_length = batch["input_ids"].shape[1]

        for k in batch:
            if k.startswith("output") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch

    def unlikelihood_loss(
        self,
        model: torch.nn.Module,
        policy_logits: torch.FloatTensor,
        policy_logps: torch.FloatTensor,
        labels: torch.LongTensor,
        score: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute the Unlikelihood loss for a batch of policy and reference model log probabilities.

        Args:
            model: The policy model used
            policy_logits: Logits returned by the policy model
            policy_logps: Log probabilities of the policy model for the responses. Shape: (batch_size,)
            labels: The demonstration data. Shape: (batch_size,)
            score: Human preference score for the response. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, rewards).
            The losses tensor contains the Unlikelihood loss for each example in the batch.
            The rewards tensors contain the rewards for the responses.
        """

        if self.sft_loss_type == "shifted_ce_loss":
            # TODO: The SFT loss calculation needs to be made more general later.
            # From modeling_llama.py. The sft_loss only compute those are demonstration data.
            # Shift so that tokens < n predict n
            loss_fct = CrossEntropyLoss()
            if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                vocab_size = model.module.config.vocab_size
            else:
                vocab_size = model.config.vocab_size

            is_positive = (score == 1)
            shift_logits_positive = policy_logits[is_positive, :-1, :].contiguous()
            shift_labels_positive = labels[is_positive, 1:].contiguous()
            # Flatten the tokens
            shift_logits_positive = shift_logits_positive.view(-1, vocab_size)
            shift_labels_positive = shift_labels_positive.view(-1)
            # Enable model parallelism
            shift_labels_positive = shift_labels_positive.to(shift_logits_positive.device)
            positive_loss = loss_fct(shift_logits_positive, shift_labels_positive)

            is_negative = (score != 1)
            shift_logits_negative = policy_logits[is_negative, :-1, :].contiguous()
            shift_labels_negative = labels[is_negative, 1:].contiguous()
            # Flatten the tokens
            shift_logits_negative = shift_logits_negative.view(-1, vocab_size)
            shift_labels_negative = shift_labels_negative.view(-1)
            # Enable model parallelism
            shift_labels_negative = shift_labels_negative.to(shift_logits_negative.device)
            negative_loss = (-1) * loss_fct(shift_logits_negative, shift_labels_negative)

        elif self.sft_loss_type == "logloss":
            positive_loss = - policy_logps[score == 1].mean()
            negative_loss = policy_logps[score != 1].mean()

        else:
            assert False, f"Unknown unlikelihood sft_loss_type: {self.sft_loss_type}"

        # Calculate Unlikelihood loss
        losses = torch.nan_to_num(positive_loss, nan=0.0) + torch.nan_to_num(negative_loss, nan=0.0)

        return losses

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities.
                    Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token.
                    Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels
                     under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not self.is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / torch.sum(loss_mask, -1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()})  # avoid error

        logits = model(
            input_ids=batch_copied["input_ids"],
            attention_mask=batch_copied["attention_mask"],
            return_dict=True
        ).logits.to(torch.float32)

        logps = self._get_batch_logps(
            logits,
            batch["labels"],
            average_log_prob=True
        )
        
        return logps, logits

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the Unlikelihood loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (policy_logps, policy_logits) = self.concatenated_forward(model, batch)

        losses = self.unlikelihood_loss(model, policy_logits, policy_logps, batch["labels"], batch["score"]) 

        logps_positive = policy_logps[batch["score"] == 1]
        logps_negative = policy_logps[batch["score"] != 1]

        logits_positive = policy_logits[batch["score"] == 1]
        logits_negative = policy_logits[batch["score"] != 1]

        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
        
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}logps/positive"] = np.nanmean(logps_positive.detach().cpu().numpy())
            metrics[f"{prefix}logps/negative"] = np.nanmean(logps_negative.detach().cpu().numpy())
            metrics[f"{prefix}logits/positive"] = np.nanmean(logits_positive.detach().cpu().numpy())
            metrics[f"{prefix}logits/negative"] = np.nanmean(logits_negative.detach().cpu().numpy())

            metrics[f"{prefix}score"] = batch["score"].detach().cpu().numpy().mean()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_unlikelihood_data_collator:
            warnings.warn(
                "compute_loss is only implemented for UnlikelihoodDataCollatorWithPadding, "
                "and you passed a datacollator that is different than "
                "UnlikelihoodDataCollatorWithPadding - you might see unexpected behavior. "
                "Alternatively, you can implement your own prediction_step method "
                "if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_unlikelihood_data_collator:
            warnings.warn(
                "prediction_step is only implemented for UnlikelihoodDataCollatorWithPadding, "
                "and you passed a datacollator that is different than "
                "UnlikelihoodDataCollatorWithPadding - you might see unexpected behavior. "
                "Alternatively, you can implement your own prediction_step method "
                "if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return loss.detach(), logits, labels

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = round(torch.tensor(metrics).nanmean().item(), 4)
        del self._stored_metrics[train_eval]
        return super().log(logs)
