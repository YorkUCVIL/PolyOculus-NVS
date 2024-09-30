
def parse_checkpoint_filename(filename):
    epoch_head = 'epoch='
    step_head = 'step='
    epoch_start = filename.find(epoch_head)+len(epoch_head)
    step_start = filename.find(step_head)+len(step_head)
    epoch = int(filename[epoch_start:epoch_start+4])
    step = int(filename[step_start:step_start+8])
    return epoch, step
