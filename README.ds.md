# Use worker

The worker uses the capture module to capture sound and hosts the model
interaction. Both model and capture run in their own respective processes
and interact by means of a queue.

To run the worker with audio input from a file put a WAV file in

	`samples/SA1.WAV`

(16k, 16 bit LE) and run `worker.py` with the `--use_file` option.

# PaddlePaddle LM

	git clone <paddlepaddle DeepSpeech repo>
	cd DeepSpeech/decoders/swig
	sh setup.sh
