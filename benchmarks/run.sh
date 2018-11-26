echo "Running 1x Shakespeare"
echo "Baseline" 
cat t8.shakespeare.txt > /dev/null
time python baseline.py t8.shakespeare.txt

echo "Dask" 
cat t8.shakespeare.txt > /dev/null
time python tf-idf-dask.py t8.shakespeare.txt 

echo "Dampr" 
cat t8.shakespeare.txt > /dev/null
time python tf-idf-dampr.py t8.shakespeare.txt 

# Run with bigger
cat t8.shakespeare.txt t8.shakespeare.txt t8.shakespeare.txt t8.shakespeare.txt > bigger.txt

echo
echo "Running 4x Shakespeare"
echo
echo "Baseline" 
cat bigger.txt > /dev/null
time python baseline.py bigger.txt

echo
echo "Dask" 
cat bigger.txt > /dev/null
time python tf-idf-dask.py bigger.txt

echo
echo "Dampr" 
cat bigger.txt > /dev/null
time python tf-idf-dampr.py bigger.txt

echo
echo "Running 20x Shakespeare"

cat bigger.txt  bigger.txt bigger.txt bigger.txt  bigger.txt > biggest.txt
mv biggest.txt bigger.txt

echo
echo "Baseline" 
cat bigger.txt > /dev/null
time python baseline.py bigger.txt

echo
echo "Dask" 
cat bigger.txt > /dev/null
time python tf-idf-dask.py bigger.txt 

echo
echo "Dampr" 
cat bigger.txt > /dev/null
time python tf-idf-dampr.py bigger.txt 

cat bigger.txt  bigger.txt bigger.txt bigger.txt  bigger.txt > biggest.txt
mv biggest.txt bigger.txt

echo "Running 500x Shakespeare"

cat bigger.txt bigger.txt bigger.txt bigger.txt bigger.txt > biggest.txt
mv biggest.txt bigger.txt

echo
echo "Baseline" 
cat bigger.txt > /dev/null
time python baseline.py bigger.txt

echo
echo "Dampr" 
cat bigger.txt > /dev/null
time python tf-idf-dampr.py bigger.txt

rm bigger.txt

