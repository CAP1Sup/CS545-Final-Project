import time
import torch
from benchmark_config import configurations

from src.benchmark_utils import (
    get_val_data,
    perform_val,
)
from src.student import StudentNetwork
from src.arcface import build_arcface_model

if __name__ == "__main__":
    # ======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg["SEED"]  # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg[
        "DATA_ROOT"
    ]  # the parent root where your train/val/test data are stored

    try:
        BACKBONE_RESUME_ROOT = cfg[
            "BACKBONE_RESUME_ROOT"
        ]  # the root to resume training from a saved checkpoint
    except KeyError:
        BACKBONE_RESUME_ROOT = None
        print(
            "No backbone resume file found. Evaluating the model with the default weights."
        )

    BACKBONE_NAME = cfg["BACKBONE_NAME"]

    INPUT_SIZE = cfg["INPUT_SIZE"]
    RGB_MEAN = cfg["RGB_MEAN"]  # for normalize inputs
    RGB_STD = cfg["RGB_STD"]
    EMBEDDING_SIZE = cfg["EMBEDDING_SIZE"]  # feature dimension
    BATCH_SIZE = cfg["BATCH_SIZE"]

    DEVICE = cfg["DEVICE"]
    MULTI_GPU = cfg["MULTI_GPU"]  # flag to use multiple GPUs
    GPU_ID = cfg["GPU_ID"]  # specify your GPU ids
    PIN_MEMORY = cfg["PIN_MEMORY"]
    NUM_WORKERS = cfg["NUM_WORKERS"]

    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    (
        lfw,
        cfp_ff,
        cfp_fp,
        agedb,
        calfw,
        cplfw,
        vgg2_fp,
        lfw_issame,
        cfp_ff_issame,
        cfp_fp_issame,
        agedb_issame,
        calfw_issame,
        cplfw_issame,
        vgg2_fp_issame,
    ) = get_val_data(DATA_ROOT)

    # Load the model
    if BACKBONE_NAME == "Custom":
        model_name = "/".join(BACKBONE_RESUME_ROOT.split("/")[-3:-1])
        BACKBONE = StudentNetwork(model_name=model_name)
    elif BACKBONE_NAME == "ArcFace":
        BACKBONE = build_arcface_model(DEVICE).get_model()
        BACKBONE_RESUME_ROOT = None  # ArcFace does not require a resume file
    else:
        raise ValueError(
            "The backbone name is not supported. Please choose from ['Custom', 'ArcFace']."
        )

    # Load pretrained weights
    if BACKBONE_RESUME_ROOT:
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))

    print("{} Backbone Generated".format(BACKBONE_NAME))

    print("=" * 60)
    print(
        "Performing evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP..."
    )
    start_time = time.time()

    print("Starting eval on LFW...")
    accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(
        MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame
    )

    print("Starting eval on CFP_FF...")
    accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(
        MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame
    )

    print("Starting eval on CFP_FP...")
    accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(
        MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame
    )

    print("Starting eval on AgeDB...")
    accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(
        MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame
    )

    print("Starting eval on CALFW...")
    accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(
        MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, calfw, calfw_issame
    )

    print("Starting eval on CPLFW...")
    accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(
        MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cplfw, cplfw_issame
    )

    print("Starting eval on VGG2_FP...")
    accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(
        MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, vgg2_fp, vgg2_fp_issame
    )

    print(
        "Evaluation: LFW Acc: {:.5f}, CFP_FF Acc: {:.5f}, CFP_FP Acc: {:.5f}, AgeDB Acc: {:.5f}, CALFW Acc: {:.5f}, CPLFW Acc: {:.5f}, VGG2_FP Acc: {:.5f}".format(
            accuracy_lfw,
            accuracy_cfp_ff,
            accuracy_cfp_fp,
            accuracy_agedb,
            accuracy_calfw,
            accuracy_cplfw,
            accuracy_vgg2_fp,
        )
    )
    print("Evaluation Time: {:.2f} seconds".format(time.time() - start_time))
