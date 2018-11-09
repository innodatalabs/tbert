from torch.optim.lr_scheduler import LambdaLR


class LineraDecayWithWarpupLR(LambdaLR):

    def __init__(self, optimizer, train_steps, warmup_steps, last_epoch=-1):

        def schedule(step):
            if step <= warmup_steps:
                return step / warmup_steps
            assert step <= train_steps
            return 1. - (train_steps - step) / (train_steps - warmup_steps)

        LambdaLR(optimizer, schedule, last_epoch=last_epoch)
