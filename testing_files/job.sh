for value in `seq 10`
do
	python3 ./tester.py
	echo $value
done	
echo All done

python3 ./tester2.py
