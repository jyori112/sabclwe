LANGS_PRETRAINED = en es fi it
LANG_PAIRS = en-es en-it en-fi en-ja

VOCABSIZE = 200000

CLDIR = data/processed/cl
EMBDIR = data/processed/wordemb

.SECONDARY:

define LearnCLWEAndEvaluate
$(CLDIR)/en-%/$(1).en.vec: $(CLDIR)/en-%/$(1)
	python vecmap/map_embeddings.py $(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.$$*.vec \
		$(CLDIR)/en-$$*/$(1).en.vec \
		$(CLDIR)/en-$$*/$(1).$$*.vec \
		--supervised $(CLDIR)/en-$$*/$(1) --cuda

$(CLDIR)/en-%/$(1).test_eval.txt: \
	$(CLDIR)/en-%/$(1).en.vec \
	data/processed/MUSE/en-%.test.txt
	python vecmap/eval_translation.py \
		$(CLDIR)/en-$$*/$(1).en.vec \
		$(CLDIR)/en-$$*/$(1).$$*.vec \
		-d data/processed/MUSE/en-$$*.test.txt --cuda \
		| cut -c 10-15,17,28-33| sed 's/ \+/ /g' | tr ' ' '\t' \
		> $$@

endef

define EvaluateByDev
$(CLDIR)/en-%/$(1).dev_eval.txt: \
	$(CLDIR)/en-%/$(1).en.vec \
	$(2)
	python vecmap/eval_translation.py \
		$(CLDIR)/en-$$*/$(1).en.vec \
		$(CLDIR)/en-$$*/$(1).$$*.vec \
		-d $(3) --cuda \
		| cut -c 10-15,17,28-33| sed 's/ \+/ /g' | tr ' ' '\t' \
		> $$@

endef

start:
	echo "\"Unsupervised Cross-lingual Word Embeddings Based on Subword Alignment\" in Proc of CICLing 2019"

clean:
	rm -rf ./data/processed
	rm -rf ./data/output
	rm -rf vecmap
	rm -rf mpaligner_0.97
	rm -rf fastText
	rm -rf wikiextractor

##############################
#	Install Tools
##############################

vecmap:
	git clone https://github.com/artetxem/vecmap.git
	cd vecmap && git checkout 585bf74c6489419682eef9aebe7a8d15f0873b6c

mpaligner_0.97:
	wget https://osdn.net/dl/mpaligner/mpaligner_0.97.tar.gz
	tar -zxvf mpaligner_0.97.tar.gz
	cd mpaligner_0.97 && make
	rm mpaligner_0.97.tar.gz

fastText:
	git clone https://github.com/facebookresearch/fastText.git
	cd fastText && git checkout 51e6738d734286251b6ad02e4fdbbcfe5b679382 && make

wikiextractor:
	git clone https://github.com/attardi/wikiextractor.git
	cd wikiextractor && \
		git checkout 2a5e6aebc030c936c7afd0c349e6826c4d02b871 && \
		python setup.py install

##############################
#	For embeddings
##############################

# Download fasttext pretrained embeddings
data/orig/wordemb/wiki.%.vec:
	mkdir -p data/orig/wordemb
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$*.vec \
		-O $@

# Download test set of MUSE dictionaries for evaluating CLWE
data/orig/MUSE/%.test.txt:
	mkdir -p data/orig/MUSE
	wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$*.5000-6500.txt \
		-O $@

data/orig/MUSE/%.train.txt:
	mkdir -p data/orig/MUSE
	wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$*.0-5000.txt \
		-O $@

data/processed/MUSE/%.test.txt: data/orig/MUSE/%.test.txt
	mkdir -p data/processed/MUSE
	cp $< $@

data/processed/MUSE/%.train.txt data/processed/MUSE/%.dev.txt: data/orig/MUSE/%.train.txt
	mkdir -p data/processed/MUSE
	python scripts/split_muse.py --train-out data/processed/MUSE/$*.train.txt \
		--dev-out data/processed/MUSE/$*.dev.txt --dev-size 500 --seed 0 \
		< $<

########## Japanese word embeddings ##########

# Download wikipedia dump on which we train Japanese word embeddings
data/orig/wiki.ja.dump.bz2:
	wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2 \
		-O $@

# Parse Japanese wiki dump
data/interim/ja/wiki.ja.dump.txt: data/orig/wiki.ja.dump.bz2 wikiextractor
	mkdir -p data/interim/ja
	python ./wikiextractor/WikiExtractor.py ./data/orig/wiki.ja.dump.bz2 -o - \
		> $@

# Tokenize Japanese wiki dump
data/interim/ja/wiki.ja.dump.tokenized: data/interim/ja/wiki.ja.dump.txt
	mecab -Owakati < $< > $@

# Train Japanese Word embeddings
data/interim/ja/wiki.ja.vec: ./data/interim/ja/wiki.ja.dump.tokenized fastText
	fastText/fasttext skipgram -input $< \
		-output data/interim/ja/wiki.ja -dim 300

########## Limit vocab size of word embeddings ##########
$(EMBDIR)/wiki.%.vec: data/orig/wordemb/wiki.%.vec
	mkdir -p $(EMBDIR)
	head -n `expr $(VOCABSIZE) + 1` $< | sed '1s/^.*$$/$(VOCABSIZE) 300/g' > $@

$(EMBDIR)/wiki.ja.vec: data/interim/ja/wiki.ja.vec
	mkdir -p $(EMBDIR)
	head -n `expr $(VOCABSIZE) + 1` $< | sed '1s/^.*$$/$(VOCABSIZE) 300/g' > $@

########## Train CLWE ##########
$(CLDIR)/en-%/unsup.en.vec: vecmap \
	data/processed/MUSE/en-%.test.txt \
	$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.%.vec
	mkdir -p $(CLDIR)/en-$*
	python vecmap/map_embeddings.py \
		$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.$*.vec \
		$(CLDIR)/en-$*/unsup.en.vec $(CLDIR)/en-$*/unsup.$*.vec \
		--unsupervised --log $(CLDIR)/en-$*/unsup.log.tsv \
		--validation data/processed/MUSE/en-$*.test.txt --cuda

########## Evaluate CLWE ##########
$(CLDIR)/en-%/unsup.test_eval.txt: $(CLDIR)/en-%/unsup.en.vec
	python vecmap/eval_translation.py $(CLDIR)/en-$*/unsup.en.vec $(CLDIR)/en-$*/unsup.$*.vec \
		-d data/processed/MUSE/en-$*.test.txt \
		| cut -c 10-15,17,28-33| sed 's/ \+/ /g' | tr ' ' '\t' \
		> $@

########## Proposing Method ##########
# Induce dictionary from CLWE
$(CLDIR)/en-%/induced_dict: $(CLDIR)/en-%/unsup.en.vec
	python induce_dict.py $(CLDIR)/en-$*/unsup.en.vec $(CLDIR)/en-$*/unsup.$*.vec --csls 10 \
		> $@

# Split induced dictionary to character levels so that it can be processed by mpaligner
$(CLDIR)/en-%/induced_dict.char: $(CLDIR)/en-%/induced_dict mpaligner_0.97 
	cat $< | cut -f1,2 | perl mpaligner_0.97/script/separate_for_char.pl > $@

# Apply mpaligner
$(CLDIR)/en-%/induced_dict.char.align: $(CLDIR)/en-%/induced_dict.char
	mpaligner_0.97/mpaligner -i $< -s || true

# Parse align result to obtain alignment score
# Ths output file is tsv with "src_word trg_word score"
$(CLDIR)/en-%/induced_dict.align_score: $(CLDIR)/en-%/induced_dict.char.align
	python parse_aligned.py < $< | sort -rnk3 > $@

# Split data in to dev and train
$(CLDIR)/en-%/induced_dict.align_score.dev: $(CLDIR)/en-%/induced_dict.align_score
	head -n 100 < $< | cut -f1,2 > $@

$(CLDIR)/en-%/induced_dict.align_score.train: $(CLDIR)/en-%/induced_dict.align_score
	tail -n +101 < $< > $@

# Create dictionary with various thresholds (Trying to find a way to make this code cleaner)
$(CLDIR)/en-%/induced_dict.align_score-2.5: $(CLDIR)/en-%/induced_dict.align_score.train
	awk '{ if ($$3 > -2.5) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.align_score.train \
		> $(CLDIR)/en-$*/induced_dict.align_score-2.5

$(CLDIR)/en-%/induced_dict.align_score-3.0: $(CLDIR)/en-%/induced_dict.align_score.train
	awk '{ if ($$3 > -3.0) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.align_score.train \
		> $(CLDIR)/en-$*/induced_dict.align_score-3.0

$(CLDIR)/en-%/induced_dict.align_score-3.5: $(CLDIR)/en-%/induced_dict.align_score.train
	awk '{ if ($$3 > -3.5) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.align_score.train \
		> $(CLDIR)/en-$*/induced_dict.align_score-3.5

$(CLDIR)/en-%/induced_dict.align_score-4.0: $(CLDIR)/en-%/induced_dict.align_score.train
	awk '{ if ($$3 > -4.0) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.align_score.train \
		> $(CLDIR)/en-$*/induced_dict.align_score-4.0

$(CLDIR)/en-%/induced_dict.align_score-4.5: $(CLDIR)/en-%/induced_dict.align_score.train
	awk '{ if ($$3 > -4.5) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.align_score.train \
		> $(CLDIR)/en-$*/induced_dict.align_score-4.5

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call LearnCLWEAndEvaluate,induced_dict.align_score$(score))))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call EvaluateByDev,induced_dict.align_score$(score),\
	$(CLDIR)/en-%/induced_dict.align_score.dev,\
	$(CLDIR)/en-$$*/induced_dict.align_score.dev)))

$(CLDIR)/en-%/induced_dict.align_score.dev_eval.txt: \
	$(CLDIR)/en-%/induced_dict.align_score-2.5.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.align_score-3.0.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.align_score-3.5.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.align_score-4.0.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.align_score-4.5.dev_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(CLDIR)/en-$*/induced_dict.align_score$$score.dev_eval.txt >> $@ ; \
	done

$(CLDIR)/en-%/induced_dict.align_score.test_eval.txt: \
	$(CLDIR)/en-%/induced_dict.align_score-2.5.test_eval.txt \
	$(CLDIR)/en-%/induced_dict.align_score-3.0.test_eval.txt \
	$(CLDIR)/en-%/induced_dict.align_score-3.5.test_eval.txt \
	$(CLDIR)/en-%/induced_dict.align_score-4.0.test_eval.txt \
	$(CLDIR)/en-%/induced_dict.align_score-4.5.test_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(CLDIR)/en-$*/induced_dict.align_score$$score.test_eval.txt >> $@ ; \
	done

$(CLDIR)/en-%/induced_dict.align_score.best.txt: \
	$(CLDIR)/en-%/induced_dict.align_score.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.align_score.test_eval.txt
	paste $(CLDIR)/en-$*/induced_dict.align_score.dev_eval.txt \
		$(CLDIR)/en-$*/induced_dict.align_score.test_eval.txt \
		| cut -f1,3,6| sort -rnk2| head -n 1| cut -f1,3 > $@

########## Unsupervised with CSLS filtering ##########
$(CLDIR)/en-%/induced_dict.csls_score: $(CLDIR)/en-%/induced_dict
	sort $^ -rnk3 > $@

$(CLDIR)/en-%/induced_dict.csls_score.dev: $(CLDIR)/en-%/induced_dict.csls_score
	head -n 100 < $< | cut -f1,2 > $@

$(CLDIR)/en-%/induced_dict.csls_score.train: $(CLDIR)/en-%/induced_dict.csls_score
	tail -n +101 < $< > $@

$(CLDIR)/en-%/induced_dict.csls_score0.9: $(CLDIR)/en-%/induced_dict.csls_score.train
	awk '{ if ($$3 > 0.9) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.csls_score.train \
		> $@

$(CLDIR)/en-%/induced_dict.csls_score0.8: $(CLDIR)/en-%/induced_dict.csls_score.train
	awk '{ if ($$3 > 0.8) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.csls_score.train \
		> $@

$(CLDIR)/en-%/induced_dict.csls_score0.7: $(CLDIR)/en-%/induced_dict.csls_score.train
	awk '{ if ($$3 > 0.7) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.csls_score.train \
		> $@

$(CLDIR)/en-%/induced_dict.csls_score0.6: $(CLDIR)/en-%/induced_dict.csls_score.train
	awk '{ if ($$3 > 0.6) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.csls_score.train \
		> $@

$(CLDIR)/en-%/induced_dict.csls_score0.5: $(CLDIR)/en-%/induced_dict.csls_score.train
	awk '{ if ($$3 > 0.5) print $$1,$$2}' < $(CLDIR)/en-$*/induced_dict.csls_score.train \
		> $@

$(foreach score,0.9 0.8 0.7 0.6 0.5,\
	$(eval $(call LearnCLWEAndEvaluate,induced_dict.csls_score$(score))))

$(foreach score,0.9 0.8 0.7 0.6 0.5,\
	$(eval $(call EvaluateByDev,induced_dict.csls_score$(score),\
	$(CLDIR)/en-%/induced_dict.csls_score.dev,\
	$(CLDIR)/en-$$*/induced_dict.csls_score.dev)))

$(CLDIR)/en-%/induced_dict.csls_score.dev_eval.txt: \
	$(CLDIR)/en-%/induced_dict.csls_score0.9.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score0.8.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score0.7.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score0.6.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score0.5.dev_eval.txt
	for score in 0.9 0.8 0.7 0.6 0.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(CLDIR)/en-$*/induced_dict.csls_score$$score.dev_eval.txt >> $@ ; \
	done

$(CLDIR)/en-%/induced_dict.csls_score.test_eval.txt: \
	$(CLDIR)/en-%/induced_dict.csls_score0.9.test_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score0.8.test_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score0.7.test_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score0.6.test_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score0.5.test_eval.txt
	for score in 0.9 0.8 0.7 0.6 0.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(CLDIR)/en-$*/induced_dict.csls_score$$score.test_eval.txt >> $@ ; \
	done

$(CLDIR)/en-%/induced_dict.csls_score.best.txt: \
	$(CLDIR)/en-%/induced_dict.csls_score.dev_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score.test_eval.txt
	paste $(CLDIR)/en-$*/induced_dict.csls_score.dev_eval.txt \
		$(CLDIR)/en-$*/induced_dict.csls_score.test_eval.txt \
		| cut -f1,3,6| sort -rnk2| head -n 1| cut -f1,3 > $@

########## Simple Supervised Baseline ##########
$(CLDIR)/en-%/muse.en.vec: \
	data/processed/MUSE/en-%.train.txt \
	$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.%.vec
	python vecmap/map_embeddings.py $(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.$*.vec \
		$(CLDIR)/en-$*/muse.en.vec $(CLDIR)/en-$*/muse.$*.vec \
		--supervised $< --cuda

$(CLDIR)/en-%/muse.test_eval.txt: \
	$(CLDIR)/en-%/muse.en.vec data/processed/MUSE/en-%.test.txt
	python vecmap/eval_translation.py \
		$(CLDIR)/en-$*/muse.en.vec $(CLDIR)/en-$*/muse.$*.vec \
		-d data/processed/MUSE/en-$*.test.txt --cuda \
		| cut -c 10-15,17,28-33| sed 's/ \+/ /g' | tr ' ' '\t' \
		> $@


########## Supervised Baseline with our filtering ##########
# Split induced dictionary to character levels so that it can be processed by mpaligner
$(CLDIR)/en-%/muse.char: data/processed/MUSE/en-%.train.txt mpaligner_0.97 
	cat $< | cut -f1,2 | perl mpaligner_0.97/script/separate_for_char.pl > $@

# Apply mpaligner
$(CLDIR)/en-%/muse.char.align: $(CLDIR)/en-%/muse.char
	mpaligner_0.97/mpaligner -i $< -s || true

# Parse align result to obtain alignment score
# Ths output file is tsv with "src_word trg_word score"
$(CLDIR)/en-%/muse.align_score: $(CLDIR)/en-%/muse.char.align
	python parse_aligned.py < $< | sort -rnk3 > $@

# Create dictionary with various thresholds (Trying to find a way to make this code cleaner)
$(CLDIR)/en-%/muse.align_score-2.5: $(CLDIR)/en-%/muse.align_score
	awk '{ if ($$3 > -2.5) print $$1,$$2}' < $(CLDIR)/en-$*/muse.align_score \
		> $(CLDIR)/en-$*/muse.align_score-2.5

$(CLDIR)/en-%/muse.align_score-3.0: $(CLDIR)/en-%/muse.align_score
	awk '{ if ($$3 > -3.0) print $$1,$$2}' < $(CLDIR)/en-$*/muse.align_score \
		> $(CLDIR)/en-$*/muse.align_score-3.0

$(CLDIR)/en-%/muse.align_score-3.5: $(CLDIR)/en-%/muse.align_score
	awk '{ if ($$3 > -3.5) print $$1,$$2}' < $(CLDIR)/en-$*/muse.align_score \
		> $(CLDIR)/en-$*/muse.align_score-3.5

$(CLDIR)/en-%/muse.align_score-4.0: $(CLDIR)/en-%/muse.align_score
	awk '{ if ($$3 > -4.0) print $$1,$$2}' < $(CLDIR)/en-$*/muse.align_score \
		> $(CLDIR)/en-$*/muse.align_score-4.0

$(CLDIR)/en-%/muse.align_score-4.5: $(CLDIR)/en-%/muse.align_score
	awk '{ if ($$3 > -4.5) print $$1,$$2}' < $(CLDIR)/en-$*/muse.align_score \
		> $(CLDIR)/en-$*/muse.align_score-4.5

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call LearnCLWEAndEvaluate,muse.align_score$(score))))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call EvaluateByDev,muse.align_score$(score),\
	data/processed/MUSE/en-%.dev.txt,\
	data/processed/MUSE/en-$$*.dev.txt)))

$(CLDIR)/en-%/muse.align_score.dev_eval.txt: \
	$(CLDIR)/en-%/muse.align_score-2.5.dev_eval.txt \
	$(CLDIR)/en-%/muse.align_score-3.0.dev_eval.txt \
	$(CLDIR)/en-%/muse.align_score-3.5.dev_eval.txt \
	$(CLDIR)/en-%/muse.align_score-4.0.dev_eval.txt \
	$(CLDIR)/en-%/muse.align_score-4.5.dev_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(CLDIR)/en-$*/muse.align_score$$score.dev_eval.txt >> $@ ; \
	done

$(CLDIR)/en-%/muse.align_score.test_eval.txt: \
	$(CLDIR)/en-%/muse.align_score-2.5.test_eval.txt \
	$(CLDIR)/en-%/muse.align_score-3.0.test_eval.txt \
	$(CLDIR)/en-%/muse.align_score-3.5.test_eval.txt \
	$(CLDIR)/en-%/muse.align_score-4.0.test_eval.txt \
	$(CLDIR)/en-%/muse.align_score-4.5.test_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(CLDIR)/en-$*/muse.align_score$$score.test_eval.txt >> $@ ; \
	done

$(CLDIR)/en-%/muse.align_score.best.txt: \
	$(CLDIR)/en-%/muse.align_score.dev_eval.txt \
	$(CLDIR)/en-%/muse.align_score.test_eval.txt
	paste $(CLDIR)/en-$*/muse.align_score.dev_eval.txt \
		$(CLDIR)/en-$*/muse.align_score.test_eval.txt \
		| cut -f1,3,6| sort -rnk2| head -n 1| cut -f1,3 > $@

########## Concat ##########
# Create dictionary with various thresholds (Trying to find a way to make this code cleaner)

define CreateConcatDict
$(CLDIR)/en-%/concat.align_score$(1): \
	$(CLDIR)/en-%/induced_dict.align_score$(1) \
	data/processed/MUSE/en-%.train.txt
	cat $(CLDIR)/en-$$*/induced_dict.align_score$(1) data/processed/MUSE/en-$$*.train.txt \
		> $(CLDIR)/en-$$*/concat.align_score$(1)

endef

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,$(eval $(call CreateConcatDict,$(score))))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call LearnCLWEAndEvaluate,concat.align_score$(score))))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call EvaluateByDev,concat.align_score$(score),\
	data/processed/MUSE/en-%.dev.txt,\
	data/processed/MUSE/en-$$*.dev.txt)))

$(CLDIR)/en-%/concat.align_score.dev_eval.txt: \
	$(CLDIR)/en-%/concat.align_score-2.5.dev_eval.txt \
	$(CLDIR)/en-%/concat.align_score-3.0.dev_eval.txt \
	$(CLDIR)/en-%/concat.align_score-3.5.dev_eval.txt \
	$(CLDIR)/en-%/concat.align_score-4.0.dev_eval.txt \
	$(CLDIR)/en-%/concat.align_score-4.5.dev_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(CLDIR)/en-$*/concat.align_score$$score.dev_eval.txt >> $@ ; \
	done

$(CLDIR)/en-%/concat.align_score.test_eval.txt: \
	$(CLDIR)/en-%/concat.align_score-2.5.test_eval.txt \
	$(CLDIR)/en-%/concat.align_score-3.0.test_eval.txt \
	$(CLDIR)/en-%/concat.align_score-3.5.test_eval.txt \
	$(CLDIR)/en-%/concat.align_score-4.0.test_eval.txt \
	$(CLDIR)/en-%/concat.align_score-4.5.test_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(CLDIR)/en-$*/concat.align_score$$score.test_eval.txt >> $@ ; \
	done

$(CLDIR)/en-%/concat.align_score.best.txt: \
	$(CLDIR)/en-%/concat.align_score.dev_eval.txt \
	$(CLDIR)/en-%/concat.align_score.test_eval.txt
	paste $(CLDIR)/en-$*/concat.align_score.dev_eval.txt \
		$(CLDIR)/en-$*/concat.align_score.test_eval.txt \
		| cut -f1,3,6| sort -rnk2| head -n 1| cut -f1,3 > $@

$(CLDIR)/en-%/results.txt: \
	$(CLDIR)/en-%/unsup.test_eval.txt \
	$(CLDIR)/en-%/induced_dict.csls_score.best.txt \
	$(CLDIR)/en-%/induced_dict.align_score.best.txt \
	$(CLDIR)/en-%/muse.test_eval.txt \
	$(CLDIR)/en-%/muse.align_score.best.txt \
	$(CLDIR)/en-%/concat.align_score.best.txt 
	echo -n "unsup\t" >> $@
	cut -f2 $(CLDIR)/en-$*/unsup.test_eval.txt >> $@
	echo -n "unsup csls\t" >> $@
	cut -f2 $(CLDIR)/en-$*/induced_dict.csls_score.best.txt >> $@
	echo -n "unsup alignment\t" >> $@
	cut -f2 $(CLDIR)/en-$*/induced_dict.align_score.best.txt >> $@
	echo -n "sup\t" >> $@
	cut -f2 $(CLDIR)/en-$*/muse.test_eval.txt >> $@
	echo -n "sup alignment\t" >> $@
	cut -f2 $(CLDIR)/en-$*/muse.align_score.best.txt >> $@
	echo -n "sup concat\t" >> $@
	cut -f2 $(CLDIR)/en-$*/concat.align_score.best.txt >> $@


