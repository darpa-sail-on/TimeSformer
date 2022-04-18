import torch
import logging

log = logging.getLogger(__name__)

CLASS_MAPPING = {'Applyingmakeup': 22,
                 'Boating': 17,
                 'Bowling': 28,
                 'Brushing': 15,
                 'Carpentry': 24,
                 'Climbing': 10,
                 'Cooking': 20,
                 'Cutting': 21,
                 'Hammering': 0,
                 'Jumping': 8,
                 'Lifting': 29,
                 'Playing brass instruments': 12,
                 'Playing percussion instruments': 13,
                 'Playing string instruments': 14,
                 'Playing wind instruments': 11,
                 'Racquet sports': 9,
                 'Riding animals': 5,
                 'Riding machine': 6,
                 'Running': 2,
                 'Skating': 18,
                 'Skiing': 7,
                 'Splitting': 4,
                 'Standing': 25,
                 'Swimming': 1,
                 'Talking': 26,
                 'Throwing': 3,
                 'Walking': 23,
                 'Washing': 19,
                 'Wrestling': 16}

def realign_logits(unaligned_logits: torch.Tensor) -> torch.Tensor:
    realigned_logits = torch.zeros([unaligned_logits.shape[0], unaligned_logits.shape[1], 30])
    for current_idx, new_idx in enumerate(CLASS_MAPPING.values()):
        realigned_logits[:, :, new_idx] = unaligned_logits[:, :, current_idx]
        log.debug(f"Mapping {list(CLASS_MAPPING.keys())[current_idx]} to {new_idx}")
    return realigned_logits
