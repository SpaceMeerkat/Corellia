rm -r ../pickle_files/*

for value in `seq 20`
do
	python3 ./MCtester.py
	echo $value
done	
echo All done

python3 ./tester2.py
