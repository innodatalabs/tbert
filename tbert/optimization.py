import torch


class LineraDecayWithWarpupLR(torch.optim.LambdaLR):

    def __init__(self, optimizer, train_steps: int, warmup_steps: int, last_epoch=-1):

        def schedule(step):
            if stem <= warmup_steps:
                return step / warmup_steps
            assert step <= train_steps
            return 1. - (train_steps - step) / (train_steps - warmup_steps)

        torch.optim.LambdaLR(optimizer, schedule, last_epoch=last_epoch)
