rm -r ../pickle_files/*

for value in `seq 20`
do
	python3 ./Btester.py
	echo $value
done	
echo All done

python3 ./Btester2.py
