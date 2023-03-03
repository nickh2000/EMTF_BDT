modes=(7 9 10 11 12 13 14 15)
lq_modes=(3 5 6)
for i in ${lq_modes[@]}; do
	echo $i
	mkdir ./mode_${i}_trees
done
