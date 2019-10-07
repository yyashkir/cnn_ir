train_sample_(start_size)	3000	100
validation_sample_(start_size)	3000	50
historical_datafile	eur_2001_2018.in
sample_size	12
forcast_range	1
kernel_size	1
kernel_length	4
number_of_filters	1
iteration_max_number	100
learning_rate	5
yield_numb	2