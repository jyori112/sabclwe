all:
	echo "Reproduce \"Unsupervised Cross-lingual Word Embeddings Based on Subword Alignment\" in Proc of CICLing 2019"

vecmap:
	git clone https://github.com/artetxem/vecmap.git

mpaligner_0.97:
	wget https://osdn.net/dl/mpaligner/mpaligner_0.97.tar.gz
	tar -zxvf mpaligner_0.97.tar.gz
	cd mpaligner_0.97 && make
	rm mpaligner_0.97.tar.gz

install_all: install_vecmap install_mpaligner

./data/orig:
	mkdir ./data/orig

./data/processed:
	mkdir ./data/processed

./data/orig/wordemb: ./data/orig
	mkdir ./data/orig/wordemb
	for lang in en es it fi tr; do \
		wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$$lang.vec \
			-O ./data/orig/wordemb/wiki.en.vec ; \
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

./data/processed/wordemb: ./data/orig/wordemb ./data/processed
	for lang in en es it fi tr; do \
		head -n 200001 ./data/orig/wordemb/wiki.$$lang.vec | sed -i '1s/^*+$/200001 300/g' > ./data/processed/wordemb/wiki.$$lang.vec \
	done

preparation: ./data/processed/wordemb

clean:
	rm -rf ./data/*
	rm -rf vecmap
	rm -rf mpaligner_0.97
