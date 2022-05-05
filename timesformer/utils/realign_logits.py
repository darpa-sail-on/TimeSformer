import torch
import logging

log = logging.getLogger(__name__)


import torch
import logging


log = logging.getLogger(__name__)


CLASS_MAPPING = {'Brushing': 15,
                 'Swimming': 1,
                 'Bowling': 28,
                 'Riding machine': 6,
                 'Lifting': 29,
                 'Throwing': 3,
                 'Cooking': 20,
                 'Applyingmakeup': 22,
                 'Walking': 23,
                 'Racquet sports': 9,
                 'Skiing': 7,
                 'Climbing': 10,
                 'Playing wind instruments': 11,
                 'Playing percussion instruments': 13,
                 'Playing string instruments': 14,
                 'Boating': 17,
                 'Skating': 18,
                 'Riding animals': 5,
                 'Jumping': 8,
                 'Washing': 19,
                 'Carpentry': 24,
                 'Playing brass instruments': 12,
                 'Wrestling': 16,
                 'Talking': 26,
                 'Splitting': 4,
                 'Cutting': 21,
                 'Hammering': 0,
                 'Standing': 25,
                 'Running': 2}

def realign_logits(unaligned_logits: torch.Tensor) -> torch.Tensor:
    realigned_logits = torch.zeros([unaligned_logits.shape[0], unaligned_logits.shape[1], 30])
    for current_idx, new_idx in enumerate(CLASS_MAPPING.values()):
        realigned_logits[:, :, new_idx] = unaligned_logits[:, :, current_idx]
        log.debug(f"Mapping {list(CLASS_MAPPING.keys())[current_idx]} to {new_idx}")
    return realigned_logits
