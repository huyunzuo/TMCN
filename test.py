import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    ppprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
import torch.backends.cudnn as cudnn
import random
import os


def setup_seed(seed=2021, deterministic=False):
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    ppprint(vars(args))

    setup_seed(args.manual_seed)
    set_gpu(args.gpu)

    trainer = FSLTrainer(args)
    trainer.evaluate_test()
    trainer.final_record()

    print(args.save_path)
