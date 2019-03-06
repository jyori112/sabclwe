all:
	echo "Reproduce \"Unsupervised Cross-lingual Word Embeddings Based on Subword Alignment\" in Proc of CICLing 2019"

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
	cd wikiextractor && git checkout 2a5e6aebc030c936c7afd0c349e6826c4d02b871 && python setup.py install

install_all: vecmap mpaligner_0.97 fastText wikiextractor

./data/orig:
	mkdir ./data/orig

./data/processed:
	mkdir ./data/processed

./data/orig/wordemb: ./data/orig
	mkdir ./data/orig/wordemb
	for lang in en es it fi tr; do \
		wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$$lang.vec \
			-O ./data/orig/wordemb/wiki.$$lang.vec ; \
	done

./data/orig/MUSE: ./data/orig
	mkdir ./data/orig/MUSE
	for lang_pair in en-es en-it en-fi en-tr; do \
		echo $$lang_pair; \
		wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$$lang_pair.0-5000.txt \
			-O ./data/orig/MUSE/$$lang_pair.train.txt; \
		wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$$lang_pair.5000-6500.txt \
			-O ./data/orig/MUSE/$$lang_pair.test.txt; \
	done

./data/orig/wiki.ja.dump.bz2: ./data/orig
	wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2 \
		-O ./data/orig/wiki.ja.dump.bz2

./data/processed/ja: ./data/processed
	mkdir ./data/processed/ja

./data/processed/ja/wiki.ja.dump.txt: ./data/processed/ja ./data/orig/wiki.ja.dump.bz2
	python ./wikiextractor/WikiExtractor.py ./data/processed/wiki.ja.dump.bz2 -O - > ./data/processed/ja/wiki.ja.dump.txt

./data/processed/ja/wiki.ja.vec: ./data/processed/ja fastText
	fastText/fasttext skipgram -input ./data/processed/ja -output ./data/processed/ja/wiki.ja.vec

./data/processed/wordemb: ./data/orig/wordemb ./data/processed ./data/processed/ja/wiki.ja.vec
	for lang in en es it fi tr; do \
		head -n 200001 ./data/orig/wordemb/wiki.$$lang.vec | sed -i '1s/^.+$/200001 300/g' \
			> ./data/processed/wordemb/wiki.$$lang.vec \
	done
	head -n 200001 ./data/processed/ja/wiki.ja.vec | sed -i '1s/^.+$/200001 300/g' \
		> ./data/processed/wordemb/wiki.$$lang.vec

setup: install_all ./data/processed/wordemb

clean:
	rm -rf ./data/*
	rm -rf vecmap
	rm -rf mpaligner_0.97

clean_except_orig:
	rm -rf ./data/processed
	rm -rf ./data/output
	rm -rf vecmap
	rm -rf mpaligner_0.97
