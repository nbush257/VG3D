#! /bin/bash
export DATA_PATH="$BOX_PATH""/__VG3D/deflection_trials/data"
for filename in "$DATA_PATH"/rat*.pkl; do 
	# echo $(basename "$filename")
	python modelling.py "$filename" --STM -C -P -w 5 --varlist M,F --plot_tgl -p 5ms_all
	python modelling.py "$filename" --STM -C -P -w 5 --varlist M,F -D --plot_tgl -p 5ms_all_deriv
done
