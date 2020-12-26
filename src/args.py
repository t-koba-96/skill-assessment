import dataclasses
import pprint
from typing import Any, Dict, Tuple

import yaml

__all__ = ["get_config"]


'''
args for train, eval
to make new yaml file, run ../runs/make_args.py
'''

# Default configs are same as "origin.yaml"
@dataclasses.dataclass(frozen=False)
class Config:

    # ============================ Dataset Size ===============================

    video_sets: str = "videos"                           # ("whole" : whole dataset, "videos" : data only which has video datas)

    # ============================= Input Features ================================

    input_feature : str = "1d"                           # ("1d" : 1d(temporal) features, "2d" : 2d(vgg16) spatial features
    input_samples : int = 400                            # input samples(len) [default : 400]

    # ============================= Spatial Attention ================================

    spatial_attention : bool = False                     # use spatial attention or not
    spatial_attention_f_maps : int = 512                 # spatial_att feature output size

    # ============================= Temporal Attention ================================

    temporal_attention_samples : int = 400               # samples for temporal attention 
    temporal_attention_size : int = 256                  # mid layer attention size (does not change feature size)
    temporal_attention_filters : int = 3                 # filters of attention layer

    # ============================= Temporal Model ================================

    temporal_model : bool = False                        # use temporal model
    num_layers: int = 9                                  # layer for each tcn stage
    num_f_maps: int = 256                                # tcn feature output size

    # ============================= Loss ================================

    diversity_loss : bool = True                         # use diversity(attention) loss
    disparity_loss : bool = True                         # use disparity(uniform compare) loss
    rank_aware_loss : bool = True                        # use rank_aware(pos neg attention) loss
    compare_loss_version : str = "v1"                    # loss version for disparity, rank_aware (v1 or v2)
    lambda_param : float = 0.1                           # weight of diversity loss
    m1 : float = 1.0                                     # margin for ranking loss
    m2 : float = 0.05                                    # margin for disparity loss
    m3 : float = 0.15                                    # margin for rank aware loss

    # =========================== Learning Configs ===============================

    epochs : int = 2000                                  # train epochs
    transform : bool = True                              # data aug (add noise to input feature)
    batch_size : int = 128                               # batch size
    lr : float = 0.0001                                  # learning rate
            
    # ============================ Runtime Configs ===============================

    workers : int = 4                                    # num of workers (dataloader)
    start_epoch : int = 1                                # start epoch for training

    # ============================ Monitor Configs ===============================

    print_freq : int = 5                                 # train console print frequency (criteria : iter)
    eval_freq : int = 10                                 # validation frequency (criteria : epoch)
    ckpt_freq : int = 5                                  # save checkpoint frequency (criteria : eval_freq)
    earlystopping : int = 20                             # earlystopping (criteria : eval_freq)


    def __post_init__(self) -> None:
        self._type_check()

        print("-" * 10, "Experiment Configuration", "-" * 10)
        pprint.pprint(dataclasses.asdict(self), width=1)

    def _type_check(self) -> None:

        _dict = dataclasses.asdict(self)

        for field, field_type in self.__annotations__.items():

            if hasattr(field_type, "__origin__"):
                # e.g.) Tuple[int].__args__[0] -> `int`
                element_type = field_type.__args__[0]

                # e.g.) Tuple[int].__origin__ -> `tuple`
                field_type = field_type.__origin__

                self._type_check_element(field, _dict[field], element_type)

            # bool is the subclass of int,
            # so need to use `type() is` instead of `isinstance`
            if type(_dict[field]) is not field_type:
                raise TypeError(
                    f"The type of '{field}' field is supposed to be {field_type}."
                )

    def _type_check_element(self, field: str, vals: Tuple[Any], element_type: type) -> None:
        for val in vals:
            if type(val) is not element_type:
                raise TypeError(
                    f"The element of '{field}' field is supposed to be {element_type}."
                )

# convert list args to tuple couse list not available in dataclass 
def convert_list2tuple(_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in _dict.items():
        if isinstance(val, list):
            _dict[key] = tuple(val)

    return _dict

# get configs from yaml file (need to be same types as Config class)
def get_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict = convert_list2tuple(config_dict)
    config = Config(**config_dict)
    return config