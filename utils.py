
def lr_scheduler(step, model_size, factor, warmup_step):
    #Zero raising to negative power is an error. Need to set step 0 to 1. Step 1 and 0 have the same lr_scheduler.
    if step == 0:
        step = 1
    lr_scheduler = factor*(model_size**(-0.5)*min(step**(-0.5), step*warmup_step**(-1.5)))
    return lr_scheduler

class Train_Counter:
    step: int = 0 #NO. of batches in the current epoch
    samples: int = 0 #NO. of obs processed
    tokens: int = 0 #NO. of total tokens processed
    
class Batchify:
    def __init__(self, text, target, padding, condition_index):
        self.x = text
        self.y = target[:, condition_index]
        self.tokens = (self.x != padding).sum()