train_sample_(start_size)	0	100
validation_sample_(start_size)	1000	100
historical_datafile	eur_2001_2018.in
sample_size	6
forcast_range	1
kernel_size	3
number_of_filters	2
iteration_max_number	1000
learning_rate	5