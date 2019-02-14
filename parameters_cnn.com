train_sample_(start_size)	3000	100
validation_sample_(start_size)	3000	50
historical_datafile	eur_2001_2018.in
sample_size	10
forcast_range	1
kernel_size	1
number_of_filters	2
iteration_max_number	2000
learning_rate	10
yield_numb	2