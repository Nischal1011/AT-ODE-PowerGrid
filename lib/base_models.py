# lib/base_models.py

"""
Shared variational objective for Latent ODE, LG-ODE, and AT-ODE.

This module is deliberately model-agnostic. It does not contain transport
logic, model selection, checkpointing, normalization, early stopping, or
authoritative evaluation/reporting metrics.
"""

import math

import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from lib.likelihood_eval import compute_mse, masked_gaussian_log_density


class VAE_Baseline(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        z0_prior,
        device,
        obsrv_std=0.01,
    ):
        super(VAE_Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        # torch.as_tensor handles Python scalars and existing tensors safely.
        self.obsrv_std = torch.as_tensor(
            obsrv_std,
            dtype=torch.float32,
            device=device,
        ).reshape(1)

        if not torch.isfinite(self.obsrv_std).all():
            raise ValueError("obsrv_std must contain only finite values.")

        if torch.any(self.obsrv_std <= 0):
            raise ValueError(
                "obsrv_std must be strictly positive; got "
                f"{self.obsrv_std.detach().cpu().tolist()}."
            )

        self.z0_prior = z0_prior

    @staticmethod
    def _validate_prediction_shapes(truth, pred_y):
        """
        Validate target and prediction tensors.

        Expected shapes:
            truth:  [n_traj, n_tp, n_dim]
            pred_y: [n_traj_samples, n_traj, n_tp, n_dim]
        """
        if not torch.is_tensor(truth):
            raise TypeError(
                f"truth must be a torch.Tensor, got {type(truth).__name__}."
            )

        if not torch.is_tensor(pred_y):
            raise TypeError(
                f"pred_y must be a torch.Tensor, got {type(pred_y).__name__}."
            )

        if truth.ndim != 3:
            raise ValueError(
                "truth must have shape [n_traj, n_tp, n_dim], but got "
                f"{tuple(truth.shape)}."
            )

        if pred_y.ndim != 4:
            raise ValueError(
                "pred_y must have shape "
                "[n_traj_samples, n_traj, n_tp, n_dim], but got "
                f"{tuple(pred_y.shape)}."
            )

        if pred_y.size(0) < 1:
            raise ValueError(
                "pred_y must contain at least one trajectory sample."
            )

        if tuple(pred_y.shape[1:]) != tuple(truth.shape):
            raise ValueError(
                "Prediction and target shapes are incompatible: "
                f"pred_y.shape[1:]={tuple(pred_y.shape[1:])}, "
                f"truth.shape={tuple(truth.shape)}."
            )

        if pred_y.device != truth.device:
            raise ValueError(
                "pred_y and truth must be on the same device; got "
                f"{pred_y.device} and {truth.device}."
            )

        if not torch.isfinite(truth).all():
            raise ValueError("truth contains NaN or infinite values.")

        if not torch.isfinite(pred_y).all():
            raise FloatingPointError(
                "pred_y contains NaN or infinite values."
            )

    @staticmethod
    def _validate_decoder_mask(mask, truth):
        """
        Validate a decoder/evaluation mask.

        Binary masks and fractional masks with values in [0, 1] are
        supported. At least one target element must be evaluated.
        """
        if not torch.is_tensor(mask):
            raise TypeError(
                f"mask must be a torch.Tensor, got {type(mask).__name__}."
            )

        if tuple(mask.shape) != tuple(truth.shape):
            raise ValueError(
                "mask must have the same shape as truth; got "
                f"mask.shape={tuple(mask.shape)} and "
                f"truth.shape={tuple(truth.shape)}."
            )

        if mask.device != truth.device:
            raise ValueError(
                "mask and truth must be on the same device; got "
                f"{mask.device} and {truth.device}."
            )

        if not torch.isfinite(mask).all():
            raise ValueError("mask contains NaN or infinite values.")

        if torch.any(mask < 0) or torch.any(mask > 1):
            raise ValueError(
                "mask must be binary or contain values within [0, 1]."
            )

        if not torch.any(mask > 0):
            raise ValueError(
                "mask contains no evaluated elements. At least one mask "
                "entry must be greater than zero."
            )

    def _expand_truth_and_mask(self, truth, pred_y, mask):
        """
        Validate and expand truth and mask across trajectory samples.

        expand() is used instead of repeat() so duplicate target tensors are
        not allocated in memory.
        """
        self._validate_prediction_shapes(truth, pred_y)

        if mask is None:
            mask = torch.ones_like(truth)

        self._validate_decoder_mask(mask, truth)

        n_traj_samples = pred_y.size(0)

        truth_expanded = truth.unsqueeze(0).expand(
            n_traj_samples,
            *truth.shape,
        )
        mask_expanded = mask.unsqueeze(0).expand(
            n_traj_samples,
            *mask.shape,
        )

        return truth_expanded, mask_expanded

    def get_gaussian_likelihood(
        self,
        truth,
        pred_y,
        temporal_weights=None,
        mask=None,
    ):
        """
        Compute the masked Gaussian reconstruction likelihood.

        Args:
            truth:
                Tensor of shape [n_traj, n_tp, n_dim].
            pred_y:
                Tensor of shape
                [n_traj_samples, n_traj, n_tp, n_dim].
            temporal_weights:
                Optional temporal weights passed to the shared likelihood
                utility.
            mask:
                Optional tensor with the same shape as truth. If None, all
                target values are evaluated.

        Returns:
            Reconstruction likelihood with shape [n_traj_samples].
        """
        truth_expanded, mask_expanded = self._expand_truth_and_mask(
            truth,
            pred_y,
            mask,
        )

        log_density_data = masked_gaussian_log_density(
            pred_y,
            truth_expanded,
            obsrv_std=self.obsrv_std,
            mask=mask_expanded,
            temporal_weights=temporal_weights,
        )
        # Shared utility returns [n_traj, n_traj_samples].
        if log_density_data.ndim != 2:
            raise ValueError(
                "masked_gaussian_log_density must return a two-dimensional "
                "tensor [n_traj, n_traj_samples], but got "
                f"{tuple(log_density_data.shape)}."
            )

        log_density_data = log_density_data.permute(1, 0)

        if not torch.isfinite(log_density_data).all():
            raise FloatingPointError(
                "Gaussian reconstruction likelihood contains NaN or "
                "infinite values. Check predictions, decoder mask coverage, "
                "and obsrv_std."
            )

        # Average over trajectories while preserving trajectory samples.
        log_density = torch.mean(log_density_data, dim=1)

        if not torch.isfinite(log_density).all():
            raise FloatingPointError(
                "Averaged reconstruction likelihood contains NaN or "
                "infinite values."
            )

        return log_density

    def get_mse(self, truth, pred_y, mask=None):
        """
        Compute masked training-space MSE.

        This is a diagnostic metric in the model's training space. Final
        normalized and inverse-normalized metrics should be calculated by
        the evaluation runner using the training-set normalizer.
        """
        truth_expanded, mask_expanded = self._expand_truth_and_mask(
            truth,
            pred_y,
            mask,
        )

        mse_values = compute_mse(
            pred_y,
            truth_expanded,
            mask=mask_expanded,
        )

        if not torch.isfinite(mse_values).all():
            raise FloatingPointError(
                "Masked MSE contains NaN or infinite values. Check model "
                "predictions and decoder mask coverage."
            )

        return torch.mean(mse_values)

    def compute_all_losses(
        self,
        batch_dict_encoder,
        batch_dict_decoder,
        batch_dict_graph,
        n_traj_samples=1,
        kl_coef=1.0,
    ):
        """
        Compute the common VAE objective.

        Latent ODE, LG-ODE, and AT-ODE all use the same objective:

            reconstruction likelihood
            - kl_coef * KL(q(z0) || p(z0))

        No model-specific or transport-specific loss is added here.
        """
        pred_y, info, temporal_weights = self.get_reconstruction(
            batch_dict_encoder,
            batch_dict_decoder,
            batch_dict_graph,
            n_traj_samples=n_traj_samples,
        )
        # pred_y:
        # [n_traj_samples, n_traj, n_tp, n_dim]

        if not isinstance(info, dict):
            raise TypeError(
                "get_reconstruction must return its second result as an "
                "information dictionary."
            )

        if "first_point" not in info:
            raise KeyError(
                "get_reconstruction info is missing 'first_point'."
            )

        first_point = info["first_point"]

        if not isinstance(first_point, (tuple, list)):
            raise TypeError(
                "info['first_point'] must be a tuple or list containing "
                "(fp_mu, fp_std, fp_enc)."
            )

        if len(first_point) != 3:
            raise ValueError(
                "info['first_point'] must contain exactly "
                "(fp_mu, fp_std, fp_enc)."
            )

        fp_mu, fp_std, fp_enc = first_point

        if not torch.is_tensor(fp_mu):
            raise TypeError("fp_mu must be a torch.Tensor.")

        if not torch.is_tensor(fp_std):
            raise TypeError("fp_std must be a torch.Tensor.")

        if fp_mu.shape != fp_std.shape:
            raise ValueError(
                "fp_mu and fp_std must have identical shapes; got "
                f"{tuple(fp_mu.shape)} and {tuple(fp_std.shape)}."
            )

        if not fp_mu.is_floating_point():
            raise TypeError(
                "fp_mu must have a floating-point dtype; got "
                f"{fp_mu.dtype}."
            )

        if not fp_std.is_floating_point():
            raise TypeError(
                "fp_std must have a floating-point dtype; got "
                f"{fp_std.dtype}."
            )

        if not torch.isfinite(fp_mu).all():
            raise FloatingPointError(
                "Posterior mean fp_mu contains NaN or infinite values."
            )

        if not torch.isfinite(fp_std).all():
            raise FloatingPointError(
                "Posterior standard deviation fp_std contains NaN or "
                "infinite values."
            )

        # Final numerical guard. The encoder should already construct the
        # standard deviation using softplus.
        fp_std = fp_std.clamp_min(1e-5)

        if not torch.isfinite(fp_std).all():
            raise FloatingPointError(
                "Posterior standard deviation contains NaN or infinite "
                "values after clamping."
            )

        fp_distr = Normal(fp_mu, fp_std)
        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if not torch.isfinite(kldiv_z0).all():
            raise FloatingPointError(
                "KL divergence contains NaN or infinite values."
            )

        if kldiv_z0.ndim < 3:
            raise ValueError(
                "Expected KL divergence with dimensions "
                "[n_traj_samples, n_traj, n_latent_dims], but got "
                f"{tuple(kldiv_z0.shape)}."
            )

        # Mean over trajectories and latent dimensions while preserving the
        # latent trajectory-sample dimension.
        kldiv_z0 = torch.mean(
            kldiv_z0,
            dim=tuple(range(1, kldiv_z0.ndim)),
        )

        if not torch.isfinite(kldiv_z0).all():
            raise FloatingPointError(
                "Reduced KL divergence contains NaN or infinite values."
            )

        if "data" not in batch_dict_decoder:
            raise KeyError(
                "batch_dict_decoder is missing the required 'data' entry."
            )

        truth = batch_dict_decoder["data"]
        decoder_mask = batch_dict_decoder.get("mask", None)

        # Validate the decoder mask before computing any loss or metric.
        self._validate_prediction_shapes(truth, pred_y)

        if decoder_mask is None:
            decoder_mask = torch.ones_like(truth)

        self._validate_decoder_mask(decoder_mask, truth)

        rec_likelihood = self.get_gaussian_likelihood(
            truth,
            pred_y,
            temporal_weights=temporal_weights,
            mask=decoder_mask,
        )

        mse = self.get_mse(
            truth,
            pred_y,
            mask=decoder_mask,
        )

        if rec_likelihood.ndim != 1:
            raise ValueError(
                "Expected rec_likelihood with shape [n_traj_samples], "
                f"but got {tuple(rec_likelihood.shape)}."
            )

        if kldiv_z0.ndim != 1:
            raise ValueError(
                "Expected reduced KL divergence with shape "
                "[n_traj_samples], but got "
                f"{tuple(kldiv_z0.shape)}."
            )

        if rec_likelihood.shape != kldiv_z0.shape:
            raise ValueError(
                "Reconstruction likelihood and KL divergence must have the "
                "same trajectory-sample shape; got "
                f"{tuple(rec_likelihood.shape)} and "
                f"{tuple(kldiv_z0.shape)}."
            )

        log_weight = rec_likelihood - kl_coef * kldiv_z0

        if log_weight.size(0) == 0:
            raise ValueError(
                "At least one trajectory sample is required."
            )

        if not torch.isfinite(log_weight).all():
            raise FloatingPointError(
                "The variational objective contains NaN or infinite "
                "log-weights."
            )

        # Normalized multi-sample importance-style objective.
        loss = -(
            torch.logsumexp(log_weight, dim=0)
            - math.log(log_weight.size(0))
        )

        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                "The computed loss contains NaN or infinite values. "
                "No silent fallback objective was applied."
            )

        results = {
            "loss": torch.mean(loss),
            "likelihood": torch.mean(rec_likelihood).detach().item(),
            "mse": torch.mean(mse).detach().item(),
            "kl_first_p": torch.mean(kldiv_z0).detach().item(),
            "std_first_p": torch.mean(fp_std).detach().item(),
        }

        return results
