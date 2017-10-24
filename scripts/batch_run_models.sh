#! /bin/bash
export DATA_PATH="$BOX_PATH""__VG3D/deflection_trials/data"
for filename in "$DATA_PATH"/rat*2017_08*D1*.pkl; do 
	# echo $(basename "$filename")
	python modelling.py "$filename" --STM -C -P -w 5 --varlist M,F -p 5ms_all
	python modelling.py "$filename" --STM -C -P -w 5 --varlist M,F -D -p 5ms_all_deriv
done
