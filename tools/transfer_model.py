import argparse
import h5py
import torch
from h5py import Group
from typing import Dict
import pickle as pkl
from collections import OrderedDict
from vast.DistributionModels.weibull import weibull as Weibull


def cmd_opts() -> argparse.Namespace:
    """
    Command line parameters for transferring model
    """
    parser = argparse.ArgumentParser("Command line for transferring model")
    parser.add_argument("--input-model", help="Path to input model")
    parser.add_argument("--output-model", help="Path to transitioned model")
    args = parser.parse_args()
    return args


def transfer_class_evm(cls_evm: Group) -> Dict:
    """
    Transfer class model to dictionary

    Args:
        cls_evm: HDF5 group containing class wise evm

    Returns:
        Dictionary with evm parameters for a class
    """
    extreme_vectors_obj = cls_evm["ExtremeVectors"]
    features = torch.tensor(cls_evm["Features"][:])
    extreme_vector_index = torch.tensor(extreme_vectors_obj["indexes"][:],
                                        dtype=torch.int64)
    scale = torch.tensor(extreme_vectors_obj["scale"])
    shape = torch.tensor(extreme_vectors_obj["shape"])
    sign_tensor = extreme_vectors_obj["sign"][()]
    translate_amount = extreme_vectors_obj["translateAmount"][()]
    small_score = torch.tensor(extreme_vectors_obj["smallScore"])
    weibull_obj = Weibull({"Scale": scale,
                           "Shape": shape,
                           "signTensor": sign_tensor,
                           "translateAmountTensor": translate_amount,
                           "smallScoreTensor": small_score})
    extreme_vectors = features[extreme_vector_index, :]
    return {
        "extreme_vectors": extreme_vectors,
        "extreme_vectors_indexes": extreme_vector_index,
        "weibulls": weibull_obj
    }


def transfer_model(input_model_path: str, output_model_path: str) -> None:
    """
    Transfer model from hdf5 to pickle

    Args:
        input_model_path: Path to input model
        output_model_path: Path to output model

    Returns:
        None
    """
    input_model = h5py.File(input_model_path, "r")
    output_model = OrderedDict()
    num_classes = len(input_model.keys())
    for i in range(1, num_classes+1):
        cls_evm = input_model[f"EVM-{i}"]
        output_model[i] = transfer_class_evm(cls_evm)
    with open(output_model_path, "wb") as f:
        pkl.dump(output_model, f)


if __name__ == "__main__":
    args = cmd_opts()
    transfer_model(args.input_model, args.output_model)
