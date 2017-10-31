import argparse
import time
import torch
from torch.autograd import Variable
from data.utils import update_progress

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seconds', type=int, default=15)
parser.add_argument('--dry_runs', type=int, default=20)
parser.add_argument('--runs', type=int, default=50)
parser.add_argument('--hidden_size', default=640, type=int)
parser.add_argument('--num_layers', default=4, type=int)
args = parser.parse_args()
hidden_size = args.hidden_size
input_standard = torch.randn(750, args.batch_size, hidden_size).cuda()  # seq_length based on max deepspeech length


class SeqLSTM(torch.nn.Sequential):
    def __init__(self, num_layers):
        super(SeqLSTM, self).__init__()
        self.rnns = [torch.nn.LSTM(hidden_size, hidden_size) for x in range(num_layers)]

    def forward(self, x):
        for rnn in self.rnns:
            x, _ = rnn(x)
        return x


model = SeqLSTM(args.num_layers)
model = model.cuda()
model.eval()


def run_benchmark(input_data):
    input_data = Variable(input_data, volatile=True)
    for n in range(args.dry_runs):
        input_data = Variable(input_data.data, volatile=True)
        model(input_data)
        update_progress(n / (float(args.dry_runs) - 1))
    print('\nDry runs finished, running benchmark')
    running_time = 0
    for n in range(args.runs):
        input_data = Variable(input_data.data, volatile=True)
        start = time.time()
        model(input_data)
        end = time.time()
        running_time += end - start
        update_progress(n / (float(args.runs) - 1))
    return running_time / float(args.runs)


print("Running standard benchmark")
run_time = run_benchmark(input_standard)
input_half = input_standard.cuda().half()
model = model.cuda().half()
print("\nRunning half precision benchmark")
run_time_half = run_benchmark(input_half)

print('\n')
print("F32 precision: Forward time %.2fs " % run_time)
print("F16 precision: Forward time %.2fs " % run_time_half)
