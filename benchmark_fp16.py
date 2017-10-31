# This script based on the parameters will benchmark the speed of F16 precision to F32 precision on GPU
import argparse
import time

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from data.utils import update_progress
from model import DeepSpeech

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seconds', type=int, default=15)
parser.add_argument('--dry_runs', type=int, default=20)
parser.add_argument('--runs', type=int, default=50)
parser.add_argument('--hidden_size', default=400, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=4, type=int, help='Number of RNN layers')
args = parser.parse_args()

input_standard = torch.randn(args.batch_size, 1, 161, args.seconds * 100).cuda()

model = DeepSpeech(rnn_hidden_size=args.hidden_size, nb_layers=args.hidden_layers, num_classes=29)
parameters = model.parameters()
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4,
                            momentum=0.9, nesterov=True)
model = torch.nn.DataParallel(model).cuda()
criterion = CTCLoss()


def iteration(input_data, cuda_half=False):
    target = torch.IntTensor(int(args.batch_size * ((args.seconds * 100) / 2))).fill_(
        1)  # targets, align half of the audio
    target_size = torch.IntTensor(args.batch_size).fill_(int((args.seconds * 100) / 2))
    input_percentages = torch.IntTensor(args.batch_size).fill_(1)

    inputs = Variable(input_data)
    target_sizes = Variable(target_size)
    targets = Variable(target)
    start = time.time()
    fwd_time = time.time()
    out = model(inputs)
    out = out.transpose(0, 1)  # TxNxH
    torch.cuda.synchronize()
    fwd_time = time.time() - fwd_time

    seq_length = out.size(0)
    sizes = Variable(input_percentages.mul_(int(seq_length)).int())
    loss_time = time.time()
    if cuda_half:
        out = out.cuda().float()
    loss = criterion(out, targets, sizes, target_sizes)
    loss = loss / inputs.size(0)  # average the loss by minibatch
    if cuda_half:
        loss = loss.cuda().half()
    loss_time = time.time() - loss_time
    bwd_time = time.time()
    # compute gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    bwd_time = time.time() - bwd_time
    end = time.time()
    return start, end, fwd_time, bwd_time, loss_time


def run_benchmark(input_data, cuda_half=False):
    for n in range(args.dry_runs):
        iteration(input_data, cuda_half)
        update_progress(n / (float(args.dry_runs) - 1))
    print('\nDry runs finished, running benchmark')
    running_time = 0
    total_fwd_time = 0
    total_bwd_time = 0
    total_loss_time = 0
    for n in range(args.runs):
        start, end, fwd_time, bwd_time, loss_time = iteration(input_data, cuda_half)
        running_time += end - start
        total_fwd_time += fwd_time
        total_bwd_time += bwd_time
        total_loss_time += loss_time
        update_progress(n / (float(args.runs) - 1))
    bwd_time = total_bwd_time / float(args.runs)
    fwd_time = total_fwd_time / float(args.runs)
    loss_time = total_loss_time / float(args.runs)
    return running_time / float(args.runs), fwd_time, bwd_time, loss_time


print("Running standard benchmark")
run_time, fwd_time, bwd_time, loss_time = run_benchmark(input_standard)
input_half = input_standard.cuda().half()
model = model.cuda().half()
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4,
                            momentum=0.9, nesterov=True)
print("\nRunning half precision benchmark")
run_time_half, fwd_time_half, bwd_time_half, loss_time_half = run_benchmark(input_half, cuda_half=True)

print('\n')
print("Average times for DeepSpeech training in seconds: ")
print("F32 precision: Average training loop %.2fs Forward: %.2fs Backward: %.2fs Loss time: %.2fs " % (
    run_time, fwd_time, bwd_time, loss_time))
print("F16 precision: Average training loop %.2fs Forward: %.2fs Backward: %.2fs Loss time: %.2fs  " % (
    run_time_half, fwd_time_half, bwd_time_half, loss_time_half))
