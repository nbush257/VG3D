#! /bin/bash
export DATA_PATH="$BOX_PATH""/__VG3D/deflection_trials/data"
for filename in "$DATA_PATH"/*.pkl; do 
	# echo $(basename "$filename")
	# python modelling.py "$filename" --STM -C -P --GLM -w 5 --varlist M,F --plot_tgl -p 5ms_all
	# python modelling.py "$filename" --STM -C -P --GLM -w 5 --varlist M,F -D --plot_tgl -p 5ms_all_deriv
	python modelling.py "$filename" --STM --GLM -w 1 --varlist M,F -D --plot_tgl -p 1ms_stm_glm_deriv --silence_noncontact
done
