
from utils.config import get_config
from utils.dataset import create_dataset_cifar10 create_dataset_imagenet

import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore import dataset as de
from mindspore import context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def modelarts_pre_process():
    pass
    # config.ckpt_path = os.path.join(config.output_path, str(get_rank_id()), config.checkpoint_path)

@moxing_wrapper(pre_process=modelarts_pre_process)


def train(config):
    # print(config)
    print('device id:', get_device_id())
    print('device num:', get_device_num())
    print('rank id:', get_rank_id())
    print('job id:', get_job_id())

    DEVICE_TARGET = config.DEVICE_TARGET
    context.set_context(mode=context.GRAPH_MODE, device_target=config.DEVICE_TARGET)
    context.set_context(save_graphs=False)
    if DEVICE_TARGET == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")

    device_num = get_device_num()
    if config.DATA.DATASET == "cifar10":
        if device_num > 1:
            config.TRAIN.BASE_LR = config.TRAIN.BASE_LR * device_num
            config.epoch_size = config.epoch_size * 2
    elif config.DATA.DATASET == "imagenet":
        pass
    else:
        raise ValueError("Unsupported dataset.")

    if device_num > 1:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, \
            parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if DEVICE_TARGET == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
        elif DEVICE_TARGET == "GPU":
            init()
    else:
        context.set_context(device_id=get_device_id())

    if config.DATA.DATASET == "cifar10":
        ds_train = create_dataset_cifar10(config, config.DATA.DATA_PATH, config.DATA.BATCH_SIZE, target=config.DEVICE_TARGET)
    elif config.DATA.DATASET == "imagenet":
        ds_train = create_dataset_imagenet(config, config.DATA.DATA_PATH, config.DATA.BATCH_SIZE)
    else:
        raise ValueError("Unsupported dataset.")

    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    #建立模型
    network = SwinTransformer(config.num_classes, phase='train')

    loss_scale_manager = None
    metrics = None
    step_per_epoch = ds_train.get_dataset_size() if config.SINK_SIZE == -1 else config.SINK_SIZE
    if config.DATA.DATASET == 'cifar10':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_cifar10(0, config.TRAIN.BASE_LR, config.TRAIN.EPOCHS, step_per_epoch))
        opt = nn.Momentum(network.trainable_params(), lr, config.momentum)
        metrics = {"Accuracy": Accuracy()}

    elif config.DATA.DATASET == 'imagenet':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_imagenet(config.TRAIN.BASE_LR, config.TRAIN.EPOCHS, step_per_epoch))
        opt = nn.Momentum(params=get_param_groups(network),
                          learning_rate=lr,
                          momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                          weight_decay=config.TRAIN.WEIGHT_DECAY,
                          loss_scale=config.LOSS_SCALE)

        from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
        if config.IS_DYNAMIC_LOSS_SCALE == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(config.LOSS_SCALE, drop_overflow_update=False)

    else:
        raise ValueError("Unsupported dataset.")

    if DEVICE_TARGET == "Ascend":
        model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2", keep_batchnorm_fp32=False,
                      loss_scale_manager=loss_scale_manager)
    elif DEVICE_TARGET == "GPU":
        model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2",
                      loss_scale_manager=loss_scale_manager)
    else:
        raise ValueError("Unsupported platform.")

    if device_num > 1:
        ckpt_save_dir = os.path.join(config.CKPT_PATH + "_" + str(get_rank()))
    else:
        ckpt_save_dir = config.CKPT_PATH

    time_cb = TimeMonitor(data_size=step_per_epoch)
    config_ck = CheckpointConfig(save_checkpoint_steps=config.SAVE_CHECKPOINT_STEPS,
                                 keep_checkpoint_max=config.KEEP_CHECKPOINT_MAX)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_alexnet", directory=ckpt_save_dir, config=config_ck)

    print("============== Starting Training ==============")
    model.train(config.TRAIN.EPOCHS, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
                dataset_sink_mode=config.DATASET_SINK_MODE, sink_size=config.SINK_SIZE)

if __name__ == "__main__":
    _, config = parse_option()
    train(config)