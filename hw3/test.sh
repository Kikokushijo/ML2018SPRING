if [ ! -f ./04170222.h5 ]; then
	wget https://gitlab.com/Kikokushijo/ML_models/raw/master/04170222.h5 -P ./
fi
python3 test.py $1 $2 $3