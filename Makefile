LANGS_PRETRAINED = en es fi it
LANG_PAIRS = en-es en-it en-fi en-ja

VOCABSIZE = 200000

CLDIR = data/processed/cl
EMBDIR = data/processed/wordemb

IPADIC_VERSION = "mecab-ipadic-2.7.0-20070801"

.SECONDARY:

all:
	echo "\"Unsupervised Cross-lingual Word Embeddings Based on Subword Alignment\" in Proc of CICLing 2019"

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
$(CLDIR)/en-%/unsup.%.vec: vecmap data/orig/MUSE/en-%.test.txt $(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.%.vec
	mkdir -p $(CLDIR)/en-$*
	python vecmap/map_embeddings.py \
		$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.$*.vec \
		$(CLDIR)/en-$*/unsup.en.vec $(CLDIR)/en-$*/unsup.$*.vec \
		--unsupervised --log $(CLDIR)/en-$*/unsup.log.tsv --validation data/orig/MUSE/en-$*.test.txt --cuda

########## Evaluate CLWE ##########
$(CLDIR)/en-%/unsup.evaluation.txt: $(CLDIR)/en-%/unsup.%.vec
	python vecmap/eval_translation.py $(CLDIR)/en-$*/unsup.en.vec $(CLDIR)/en-$*/unsup.$*.vec -d data/orig/MUSE/en-$*.test.txt \
		> $@


$(CLDIR)/en-%/induced_dict: $(CLDIR)/en-%/unsup.en.vec $(CLDIR)/en-%/unsup.%.vec
	python induced_dict.py $^ --csls 10 | sort -rnk3 > $@

$(CLDIR)/en-%/induced_dict.char: $(CLDIR)/en-%/induced_dict
	cat $< | cut -d' ' -f1,2 | perl mpaligner_0.97/script/separate_for_char.pl > $@

$(CLDIR)/en-%/induced_dict.char.align: $(CLDIR)/en-%/induced_dict.char
	mpaligner_0.97/mpaligner -i $< -o $@ -s

$(CLDIR)/en-%/induced_dict.align_score: $(CLDIR)/en-%/induced_dict.char.align
	python cli.py format-align < $< > $@

$(CLDIR)/en-%/induced_dict.dev: $(CLDIR)/en-%/induced_dict.align_score
	head -n 100 < $< > $@

$(CLDIR)/en-%/induced_dict.train: $(CLDIR)/en-%/induced_dict.align_score
	tail -n +101 < $< > $@

#$(CLDIR)/en-%/induced_dict.min_score-3.5.log: $(CLDIR)/en-%/induced_dict.train \
#	vecmap $(CLDIR)/en-%/unsup.en.vec $(CLDIR)-%/unsup.%.vec
#	cat $< | awk 'if $3 > $(1) { print $1,$2 }' |
#	python vecmap/map_embeddings.py \
#		$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.$*.vec \
#		$(CLDIR)/en-$*/induced_dict.min-score-3.5.en $(CLDIR)/en-$*/induced_dict.min-score-3.5.$* \
#		--supervised --log $@ --validation data/orig/MUSE/en-$*.test.txt --cuda
#
#$(CLDIR)/en-%/induced_dict.min_score-3.5.en: $(CLDIR)/en-%/induced_dict.min-score-3.5.log
#
#$(CLDIR)/en-%/induced_dict.min_score-3.5.%: $(CLDIR)/en-%/induced_dict.min-score-3.5.log

test:
	echo "Hello World"

clean: clean_except_orig
	rm -rf ./data/*

clean_except_orig:
	rm -rf ./data/processed
	rm -rf ./data/output
	rm -rf vecmap
	rm -rf mpaligner_0.97
	rm -rf fastText
	rm -rf wikiextractor
	rm -rf mecab
