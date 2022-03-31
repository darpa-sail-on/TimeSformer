import torch
import copy
import logging


log = logging.getLogger(__name__)


def compute_kl_divergence(mu, sigma, m, s):
    """
    Compute Kullback Leibler with Gaussian assumption of training data
    mu: mean of test batch
    sigm: standard deviation of test batch
    m: mean of all data in training data set
    s: standard deviation of all data in training data set
    return: KL distance, non negative double precison float
    """
    kl = torch.log(s/sigma) + (((sigma ** 2) + ((mu - m) ** 2)) / (2 * (s ** 2))) - 0.5
    return kl


@torch.no_grad()
def kl_divergence_based_wd(
        mean_max_probs,
        current_window,
        window_size,
        mu_train,
        sigma_train,
        kl_threshold,
        acc):
    """
    Detect Change in World using KL Divergence.

    :param mean_max_probs (torch.Tensor): Mean max probability for samples in the current round.
    :param current_window (list): Sliding window over previous samples
    :param window_size (int): Size of the window
    :param mu_train (float): Mean of the training data
    :param sigma_train (float): Variance of the training data
    :param kl_threshold (float): KL divergence threshold

    :return Predictions if the world has changed
    """
    num_samples = mean_max_probs.shape[0]

    # Populate sliding window
    previous_window = copy.deepcopy(current_window)
    current_window.extend(mean_max_probs)

    if len(current_window) >= window_size:
        current_window_size = len(current_window)
        current_window = current_window[current_window_size-window_size:]

    if len(current_window) < window_size:
        return [0.0]*num_samples, current_window
    else:
        past_p = torch.Tensor(previous_window)
        current_p = torch.Tensor(current_window)
        temp_world_changed = torch.zeros(num_samples)
        p_past_and_current = torch.cat((past_p[1:],
                                        current_p))
        p_window = p_past_and_current.unfold(0, window_size, 1)
        mu = torch.mean(p_window, dim=1)
        sigma = torch.std(p_window, dim=1)
        kl_epoch = compute_kl_divergence(mu, sigma, mu_train, sigma_train)
        log.info(f"max kl_epoch = {torch.max(kl_epoch)}")
        W = (kl_epoch / (2 * kl_threshold))
        log.info(f"W = {W.tolist()}")
        W[0] = torch.clamp(W[0], min=acc)
        W , _ = torch.cummax (W, dim=0)
        temp_world_changed = torch.clamp(W, max=1.0)[len(W) - num_samples:]
        temp_world_changed = list(torch.clamp(temp_world_changed, min=0).detach().numpy())
        log.info(f"temp_world_changed = {temp_world_changed}")
        return temp_world_changed, current_window
