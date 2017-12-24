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

You also need a vocabulary (word list) and the PP language model. You
can get the language model here:

	cd DeepSpeech/models/lm/
	sh download_lm_en.sh

To start the worker with the PP language model:

	python worker.py \
		--decoder pp \
		--scorer_vocab_path alphabet.txt \
		--lm_path pp/DeepSpeech/models/lm/common_crawl_00.prune01111.trie.klm

where alphabet.txt contains the alphabet the acoustic model was trained
on. Note that the case of the letters needs to match the ones the
LM was trained with, in case of PP lower-case ones, to actually yield
results.
