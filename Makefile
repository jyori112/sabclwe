DATA=data
INPUT=$(DATA)/input
INTERM=$(DATA)/interm
OUTPUT=$(DATA)/output
RESULT=$(DATA)/result
ANALYSIS=$(DATA)/analysis

CLDIR=$(OUTPUT)/cl
EMBDIR=$(OUTPUT)/wordemb

##################################################
#	Configuration
##################################################
VOCABSIZE = 200000

.SECONDARY:

define LearnCLWEAndEvaluate
$(CLDIR)/en-%/$(1).en.vec: $(CLDIR)/en-%/$(1)
	python vecmap/map_embeddings.py $(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.$$*.vec \
		$(CLDIR)/en-$$*/$(1).en.vec \
		$(CLDIR)/en-$$*/$(1).$$*.vec \
		--supervised $(CLDIR)/en-$$*/$(1) --cuda

$(RESULT)/cl/en-%/$(1).test_eval.txt:\
	$(CLDIR)/en-%/$(1).en.vec \
	$(OUTPUT)/MUSE/en-%.test.txt
	python vecmap/eval_translation.py \
		$(CLDIR)/en-$$*/$(1).en.vec \
		$(CLDIR)/en-$$*/$(1).$$*.vec \
		-d $(OUTPUT)/MUSE/en-$$*.test.txt --cuda \
		| python format_eval.py > $$@

endef

define EvaluateByDev
$(RESULT)/cl/en-%/$(1).dev_eval.txt:\
	$(CLDIR)/en-%/$(1).en.vec \
	$(2)
	python vecmap/eval_translation.py \
		$(CLDIR)/en-$$*/$(1).en.vec \
		$(CLDIR)/en-$$*/$(1).$$*.vec \
		-d $(3) --cuda \
		| python format_eval.py > $$@

endef

define FilterDict
$(CLDIR)/en-%/$(2): $(3)
	awk '{ if ($$$$3 > $(1)) print $$$$1,$$$$2}' < $$< \
		> $$@
	if [ ! -s $$@ ]; then \
		head $$< | cut -f1,2 > $$@ ; \
	fi

endef

start:
	echo "\"Unsupervised Cross-lingual Word Embeddings Based on Subword Alignment\" in Proc of CICLing 2019"

clean:
	rm -rf ./$(OUTPUT)
	rm -rf ./data/output
	rm -rf ./graphs
	rm -rf ./analysis
	rm -rf mpaligner_0.97
	rm -rf fastText
	rm -rf wikiextractor

##############################
#	Install Tools
##############################
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
$(INPUT)/wordemb/wiki.%.vec:
	mkdir -p $$(dirname $@)
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$*.vec \
		-O $@

# Download test set of MUSE dictionaries for evaluating CLWE
$(INPUT)/MUSE/%.test.txt:
	mkdir -p $$(dirname $@)
	wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$*.5000-6500.txt \
		-O $@

$(INPUT)/MUSE/%.train.txt:
	mkdir -p $$(dirname $@)
	wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$*.0-5000.txt \
		-O $@

$(OUTPUT)/MUSE/%.test.txt: $(INPUT)/MUSE/%.test.txt
	mkdir -p $$(dirname $@)
	cp $< $@

$(OUTPUT)/MUSE/%.train.txt $(OUTPUT)/MUSE/%.dev.txt: $(INPUT)/MUSE/%.train.txt
	mkdir -p $$(dirname $@)
	python scripts/split_muse.py --train-out $(OUTPUT)/MUSE/$*.train.txt \
		--dev-out $(OUTPUT)/MUSE/$*.dev.txt --dev-size 500 --seed 0 \
		< $<

########## Japanese word embeddings ##########

# Download wikipedia dump on which we train Japanese word embeddings
$(INPUT)/wiki.ja.dump.bz2:
	wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2 \
		-O $@

# Parse Japanese wiki dump
$(INTERM)/ja/wiki.ja.dump.txt: $(INPUT)/wiki.ja.dump.bz2 wikiextractor
	mkdir -p $$(dirname $@)
	python ./wikiextractor/WikiExtractor.py ./$(INPUT)/wiki.ja.dump.bz2 -o - \
		> $@

# Tokenize Japanese wiki dump
$(INTERM)/ja/wiki.ja.dump.tokenized: $(INTERM)/ja/wiki.ja.dump.txt
	mecab -Owakati < $< > $@

# Train Japanese Word embeddings
$(INTERM)/ja/wiki.ja.vec: ./$(INTERM)/ja/wiki.ja.dump.tokenized fastText
	fastText/fasttext skipgram -input $< \
		-output $(INTERM)/ja/wiki.ja -dim 300

########## Limit vocab size of word embeddings ##########
$(EMBDIR)/wiki.%.vec: $(INPUT)/wordemb/wiki.%.vec
	mkdir -p $$(dirname $@)
	head -n `expr $(VOCABSIZE) + 1` $< | sed '1s/^.*$$/$(VOCABSIZE) 300/g' > $@

$(EMBDIR)/wiki.ja.vec: $(INTERM)/ja/wiki.ja.vec
	mkdir -p $$(dirname $@)
	head -n `expr $(VOCABSIZE) + 1` $< | sed '1s/^.*$$/$(VOCABSIZE) 300/g' > $@

########## Train CLWE ##########
$(CLDIR)/en-%/unsup.en.vec: vecmap \
	$(OUTPUT)/MUSE/en-%.test.txt \
	$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.%.vec
	mkdir -p $$(dirname $@)
	python vecmap/map_embeddings.py \
		$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.$*.vec \
		$(CLDIR)/en-$*/unsup.en.vec $(CLDIR)/en-$*/unsup.$*.vec \
		--unsupervised --log $(CLDIR)/en-$*/unsup.log.tsv \
		--validation $(OUTPUT)/MUSE/en-$*.test.txt --cuda

########## Evaluate CLWE ##########
$(RESULT)/cl/en-%/unsup.test_eval.txt: $(CLDIR)/en-%/unsup.en.vec
	mkdir -p $$(dirname $@)
	python vecmap/eval_translation.py $(CLDIR)/en-$*/unsup.en.vec $(CLDIR)/en-$*/unsup.$*.vec \
		-d $(OUTPUT)/MUSE/en-$*.test.txt \
		| python format_eval.py > $@


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
$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,$(eval $(call FilterDict,\
	$(score),induced_dict.align_score$(score),$(CLDIR)/en-%/induced_dict.align_score.train)))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call LearnCLWEAndEvaluate,induced_dict.align_score$(score))))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call EvaluateByDev,induced_dict.align_score$(score),\
	$(CLDIR)/en-%/induced_dict.align_score.dev,\
	$(CLDIR)/en-$$*/induced_dict.align_score.dev)))

$(RESULT)/cl/en-%/induced_dict.align_score.dev_eval.txt:\
	$(RESULT)/cl/en-%/induced_dict.align_score-2.5.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score-3.0.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score-3.5.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score-4.0.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score-4.5.dev_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(RESULT)/cl/en-$*/induced_dict.align_score$$score.dev_eval.txt >> $@ ; \
	done

$(RESULT)/cl/en-%/induced_dict.align_score.test_eval.txt: \
	$(RESULT)/cl/en-%/induced_dict.align_score-2.5.test_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score-3.0.test_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score-3.5.test_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score-4.0.test_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score-4.5.test_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		paste \
			$(RESULT)/cl/en-$*/induced_dict.align_score$$score.test_eval.txt \
			>> $@ ; \
	done

$(RESULT)/cl/en-%/induced_dict.align_score.best.txt: \
	$(RESULT)/cl/en-%/induced_dict.align_score.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score.test_eval.txt
	paste $(RESULT)/cl/en-$*/induced_dict.align_score.dev_eval.txt \
		$(RESULT)/cl/en-$*/induced_dict.align_score.test_eval.txt \
		| cut -f1,3,6,7| sort -k2,2rn -k1,1n| head -n 1| cut -f1,3 > $@

########## Unsupervised with CSLS filtering ##########
$(CLDIR)/en-%/induced_dict.csls_score: $(CLDIR)/en-%/induced_dict
	sort $^ -rnk3 > $@

$(CLDIR)/en-%/induced_dict.csls_score.dev: $(CLDIR)/en-%/induced_dict.csls_score
	head -n 100 < $< | cut -f1,2 > $@

$(CLDIR)/en-%/induced_dict.csls_score.train: $(CLDIR)/en-%/induced_dict.csls_score
	tail -n +101 < $< > $@

$(foreach score,0.9 0.8 0.7 0.6 0.5,$(eval $(call FilterDict,\
	$(score),induced_dict.csls_score$(score),$(CLDIR)/en-%/induced_dict.csls_score.train)))

$(foreach score,0.9 0.8 0.7 0.6 0.5,\
	$(eval $(call LearnCLWEAndEvaluate,induced_dict.csls_score$(score))))

$(foreach score,0.9 0.8 0.7 0.6 0.5,\
	$(eval $(call EvaluateByDev,induced_dict.csls_score$(score),\
	$(CLDIR)/en-%/induced_dict.csls_score.dev,\
	$(CLDIR)/en-$$*/induced_dict.csls_score.dev)))

$(RESULT)/cl/en-%/induced_dict.csls_score.dev_eval.txt: \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.9.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.8.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.7.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.6.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.5.dev_eval.txt
	for score in 0.5 0.6 0.7 0.8 0.9; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(RESULT)/cl/en-$*/induced_dict.csls_score$$score.dev_eval.txt >> $@ ; \
	done

$(RESULT)/cl/en-%/induced_dict.csls_score.test_eval.txt: \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.9.test_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.8.test_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.7.test_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.6.test_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score0.5.test_eval.txt
	for score in 0.5 0.6 0.7 0.8 0.9; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(RESULT)/cl/en-$*/induced_dict.csls_score$$score.test_eval.txt >> $@ ; \
	done

$(RESULT)/cl/en-%/induced_dict.csls_score.best.txt: \
	$(RESULT)/cl/en-%/induced_dict.csls_score.dev_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score.test_eval.txt
	paste $(RESULT)/cl/en-$*/induced_dict.csls_score.dev_eval.txt \
		$(RESULT)/cl/en-$*/induced_dict.csls_score.test_eval.txt \
		| cut -f1,3,6,7| sort -k2,2rn -k1,1n| head -n 1| cut -f1,3 > $@

########## Simple Supervised Baseline ##########
$(CLDIR)/en-%/muse.en.vec: \
	$(OUTPUT)/MUSE/en-%.train.txt \
	$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.%.vec
	python vecmap/map_embeddings.py $(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.$*.vec \
		$(CLDIR)/en-$*/muse.en.vec $(CLDIR)/en-$*/muse.$*.vec \
		--supervised $< --cuda

$(RESULT)/cl/en-%/muse.test_eval.txt: \
	$(CLDIR)/en-%/muse.en.vec $(OUTPUT)/MUSE/en-%.test.txt
	python vecmap/eval_translation.py \
		$(CLDIR)/en-$*/muse.en.vec $(CLDIR)/en-$*/muse.$*.vec \
		-d $(OUTPUT)/MUSE/en-$*.test.txt --cuda \
		| python format_eval.py > $@

########## Supervised Baseline with our filtering ##########
# Split induced dictionary to character levels so that it can be processed by mpaligner
$(CLDIR)/en-%/muse.char: $(OUTPUT)/MUSE/en-%.train.txt mpaligner_0.97 
	cat $< | cut -f1,2 | perl mpaligner_0.97/script/separate_for_char.pl > $@

# Apply mpaligner
$(CLDIR)/en-%/muse.char.align: $(CLDIR)/en-%/muse.char
	mpaligner_0.97/mpaligner -i $< -s || true

# Parse align result to obtain alignment score
# Ths output file is tsv with "src_word trg_word score"
$(CLDIR)/en-%/muse.align_score: $(CLDIR)/en-%/muse.char.align
	python parse_aligned.py < $< | sort -rnk3 > $@

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,$(eval $(call FilterDict,\
	$(score),muse.align_score$(score),$(CLDIR)/en-%/muse.align_score)))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call LearnCLWEAndEvaluate,muse.align_score$(score))))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call EvaluateByDev,muse.align_score$(score),\
	$(OUTPUT)/MUSE/en-%.dev.txt,\
	$(OUTPUT)/MUSE/en-$$*.dev.txt)))

$(RESULT)/cl/en-%/muse.align_score.dev_eval.txt: \
	$(RESULT)/cl/en-%/muse.align_score-2.5.dev_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score-3.0.dev_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score-3.5.dev_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score-4.0.dev_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score-4.5.dev_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(RESULT)/cl/en-$*/muse.align_score$$score.dev_eval.txt >> $@ ; \
	done

$(RESULT)/cl/en-%/muse.align_score.test_eval.txt: \
	$(RESULT)/cl/en-%/muse.align_score-2.5.test_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score-3.0.test_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score-3.5.test_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score-4.0.test_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score-4.5.test_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(RESULT)/cl/en-$*/muse.align_score$$score.test_eval.txt >> $@ ; \
	done

$(RESULT)/cl/en-%/muse.align_score.best.txt: \
	$(RESULT)/cl/en-%/muse.align_score.dev_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score.test_eval.txt
	paste $(RESULT)/cl/en-$*/muse.align_score.dev_eval.txt \
		$(RESULT)/cl/en-$*/muse.align_score.test_eval.txt \
		| cut -f1,3,6,7| sort -k2,2rn -k1,1n| head -n 1| cut -f1,3 > $@

########## Concat ##########
# Create dictionary with various thresholds (Trying to find a way to make this code cleaner)

define CreateConcatDict
$(CLDIR)/en-%/concat.align_score$(1): \
	$(CLDIR)/en-%/induced_dict.align_score$(1) \
	$(OUTPUT)/MUSE/en-%.train.txt
	cat $(CLDIR)/en-$$*/induced_dict.align_score$(1) $(OUTPUT)/MUSE/en-$$*.train.txt \
		> $(CLDIR)/en-$$*/concat.align_score$(1)

endef

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,$(eval $(call CreateConcatDict,$(score))))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call LearnCLWEAndEvaluate,concat.align_score$(score),muse.test_eval.prediction.npy)))

$(foreach score,-2.5 -3.0 -3.5 -4.0 -4.5,\
	$(eval $(call EvaluateByDev,concat.align_score$(score),\
	$(OUTPUT)/MUSE/en-%.dev.txt,\
	$(OUTPUT)/MUSE/en-$$*.dev.txt)))

$(RESULT)/cl/en-%/concat.align_score.dev_eval.txt: \
	$(RESULT)/cl/en-%/concat.align_score-2.5.dev_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-3.0.dev_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-3.5.dev_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-4.0.dev_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-4.5.dev_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(RESULT)/cl/en-$*/concat.align_score$$score.dev_eval.txt >> $@ ; \
	done

$(RESULT)/cl/en-%/concat.align_score.test_eval.txt: \
	$(RESULT)/cl/en-%/concat.align_score-2.5.test_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-3.0.test_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-3.5.test_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-4.0.test_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-4.5.test_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-2.5.test_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-3.0.test_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-3.5.test_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-4.0.test_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score-4.5.test_eval.txt
	for score in -2.5 -3.0 -3.5 -4.0 -4.5; do \
		echo -n "$$score\t" >> $@ ; \
		cat $(RESULT)/cl/en-$*/concat.align_score$$score.test_eval.txt >> $@ ; \
	done

$(RESULT)/cl/en-%/concat.align_score.best.txt: \
	$(RESULT)/cl/en-%/concat.align_score.dev_eval.txt \
	$(RESULT)/cl/en-%/concat.align_score.test_eval.txt
	paste $(RESULT)/cl/en-$*/concat.align_score.dev_eval.txt \
		$(RESULT)/cl/en-$*/concat.align_score.test_eval.txt \
		| cut -f1,3,6,7| sort -k2,2rn -k1,1n| head -n 1| cut -f1,3 > $@

$(RESULT)/cl/en-%/results.txt: \
	$(RESULT)/cl/en-%/unsup.test_eval.txt \
	$(RESULT)/cl/en-%/induced_dict.csls_score.best.txt \
	$(RESULT)/cl/en-%/induced_dict.align_score.best.txt \
	$(RESULT)/cl/en-%/muse.test_eval.txt \
	$(RESULT)/cl/en-%/muse.align_score.best.txt \
	$(RESULT)/cl/en-%/concat.align_score.best.txt 
	echo -n "unsup\t" >> $@
	cut -f2 $(RESULT)/cl/en-$*/unsup.test_eval.txt >> $@
	echo -n "unsup csls\t" >> $@
	cut -f2 $(RESULT)/cl/en-$*/induced_dict.csls_score.best.txt >> $@
	echo -n "unsup alignment\t" >> $@
	cut -f2 $(RESULT)/cl/en-$*/induced_dict.align_score.best.txt >> $@
	echo -n "sup\t" >> $@
	cut -f2 $(RESULT)/cl/en-$*/muse.test_eval.txt >> $@
	echo -n "sup alignment\t" >> $@
	cut -f2 $(RESULT)/cl/en-$*/muse.align_score.best.txt >> $@
	echo -n "sup concat\t" >> $@
	cut -f2 $(RESULT)/cl/en-$*/concat.align_score.best.txt >> $@

##############################
#  Graph
##############################
$(ANALYSIS)/cl/en-%/sensitivity_align.pdf: \
	$(RESULT)/cl/en-%/induced_dict.align_score.test_eval.txt \
	$(RESULT)/cl/en-%/unsup.test_eval.txt 
	mkdir -p $$(dirname $@)
	python graph.py sensitivity-align $(RESULT)/cl/en-$*/induced_dict.align_score.test_eval.txt $@ \
		--baseline $(RESULT)/cl/en-$*/unsup.test_eval.txt

$(ANALYSIS)/cl/en-%/sensitivity_csls.pdf: \
	$(RESULT)/cl/en-%/induced_dict.csls_score.test_eval.txt \
	$(RESULT)/cl/en-%/unsup.test_eval.txt 
	mkdir -p $$(dirname $@)
	python graph.py sensitivity-csls $(RESULT)/cl/en-$*/induced_dict.csls_score.test_eval.txt $@ \
		--baseline $(RESULT)/cl/en-$*/unsup.test_eval.txt


##############################
#  Analysis
##############################
$(ANALYSIS)/cl/en-%/induced_dict.tsv: $(CLDIR)/en-%/induced_dict.align_score
	mkdir -p $$(dirname $@)
	cat $< | awk 'BEGIN { OFS="\t" } { if ($$1 != $$2) print NR,$$1,$$2 }'| head > $@
