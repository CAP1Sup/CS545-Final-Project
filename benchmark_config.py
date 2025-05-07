import torch


configurations = {
    1: dict(
        # Model settings
        BACKBONE_NAME="Custom",  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152', 'Custom']
        BACKBONE_RESUME_ROOT="./results/timm/tinynet_a/SynthDistill-TinyFaR-A.pt",
        # BACKBONE_RESUME_ROOT="./results/timm/tinynet_b/SynthDistill-TinyFaR-B.pt",
        # BACKBONE_RESUME_ROOT="./results/timm/tinynet_c/SynthDistill-TinyFaR-C.pt",
        # BACKBONE_RESUME_ROOT="./results/timm/resnet50.a1_in1k/epoch_9.pt",
        # BACKBONE_RESUME_ROOT="./results/timm/resnext50_32x4d.a1h_in1k/epoch_14.pt",
        # BACKBONE_NAME="ArcFace",
        #
        #
        # Evaluation settings
        # These should not need to be changed
        SEED=1337,  # random seed for reproduce results
        DATA_ROOT="./datasets",  # the parent root where your train/val/test data are stored
        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=512,  # 512
        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,  # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0, 1, 2, 3],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=0,
    ),
}
