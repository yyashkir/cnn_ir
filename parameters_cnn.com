train_sample_(start_size)	0	4000
validation_sample_(start_size)	1500	1000
historical_datafile	eur_2001_2018.in
sample_size	8
forcast_range	1
kernel_size	2
number_of_filters	7
iteration_max_number	1500
learning_rate	5
yield_numb	3