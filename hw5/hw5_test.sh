if [ ! -f ./word2vec_all_0606.model.syn1neg.npy ]; then
	wget https://gitlab.com/Kikokushijo/ML_models/raw/master/word2vec_all_0606.model.syn1neg.npy -P ./
fi
if [ ! -f ./word2vec_all_0606.model.wv.syn0.npy ]; then
	wget https://gitlab.com/Kikokushijo/ML_models/raw/master/word2vec_all_0606.model.wv.syn0.npy -P ./
fi
python3 test.py $1 $2 $3
