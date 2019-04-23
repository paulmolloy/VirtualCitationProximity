# Script for finding CPIs for a given wiki title.
for i in {1..80};do
    echo $i;
    grep "Michael Collins (Irish leader)" $i;
done
