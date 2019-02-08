//	Copyright © 2018 Yashkir Consulting
/*
04/02/2019	-	 started
				
  /  /2019	-	end
*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <sstream>
#include <cstdio>
#include <ctime>
#include "functions_uni.h"
#include <armadillo>
#include <windows.h>
using namespace std;
//
// global variables and constants
string s;

arma::Row <int> tenors;		//tenors (in days)  
arma::Mat <double> historical_dataset;		// input hist data rates for all tenors over period of time
string historical_datafile;	// file with historical yield rates 
arma::Mat <double> train_dataset;		// input hist data rates for all tenor over period of time
arma::Mat <double> validation_dataset;		// input hist data rates for all tenor over period of time

double pi = 3.14159265359;
int i_start_train;		// start day from historical yield data for training set
int train_size;			// train size
int i_start_validation;	// start day from historical yield data for validation set
int validation_size;
int N;	// yield length (number of columns in data file)
int M;	// sample size	(number of rows == days)
int Mf;	// filter length
int Nf;	// filter width
int V;	// vectorized_concatenated set of filters transformed to vector (length)
int i_hist;			// total number of lines in file with historical yields
int	forcast_range;	// number of days forward for yield forcast
int kernel_size;
int Q;				// number of filters
int  I;				// number of iterations
double h;			// learning rate
double h_start;
clock_t start;
int i_minValErr;
double minValErr = 0;
//
// reading data from input files:
void  read_input_files(char *argv)
{
	int i, k;
	//
	ifstream infile(argv); //reading control parameters from file argv
	if (!infile.is_open())
	{
		string err = "\nfile ";
		err.append(argv);
		err.append(" cannot be opened");
		cout << err;
		exit(0);
	}
	infile
		>> s >> i_start_train >> train_size
		>> s >> i_start_validation >> validation_size
		>> s >> historical_datafile
		>> s >> M
		>> s >> forcast_range
		>> s >> kernel_size
		>> s >> Q
		>> s >> I
		>> s >> h
		;
	h_start = h;
	infile.close();
	//
		//reading historical rates 
	ifstream data_stream(historical_datafile.c_str());
	data_stream >> N;		// number of tenors (=columns)
	tenors.set_size(N);
	data_stream >> i_hist;	// number of lines in file with historical yields
	for (k = 0; k < N; k++)
		data_stream >> tenors(k);
	//
	historical_dataset.set_size(i_hist, N);
	for (i = 0; i < i_hist; i++)
		for (k = 0; k < N; k++)
			data_stream >> historical_dataset(i, k);
	data_stream.close();
	cout << "\nHistorical_dataset (number of rows): " << historical_dataset.n_rows << endl;
	//
	train_size = min(train_size, i_hist - i_start_train);
	train_dataset.set_size(train_size, N);
	for (i = i_start_train; i < i_start_train + train_size; i++)
		for (k = 0; k < N; k++)
			train_dataset(i - i_start_train, k) = historical_dataset(i, k);
	//
	validation_size = min(validation_size, i_hist - i_start_validation);
	validation_dataset.set_size(validation_size, N);
	for (i = i_start_validation; i < i_start_validation + validation_size; i++)
		for (k = 0; k < N; k++)
			validation_dataset(i - i_start_validation, k) = historical_dataset(i, k);
	//
	stringstream st, sv;
	st << "Training set: " << train_dataset.n_rows << " rows";
	sv << "Validation set: " << validation_dataset.n_rows << " rows";
	cout << endl << st.str() << endl << sv.str() << endl;
//
	Mf = M - kernel_size + 1;	// filter length
	Nf = N - kernel_size + 1;	// filter width
	V = Q * Mf * Nf;			// long vector length
}
//
void iteration_loop()
{
	int it;
	int s;
	int i, j;
	int k, p;
	double sum, sum_p, sum_mn;
	int q, m, n;
	int alpha, beta;
	ofstream out_obj_fn;
	out_obj_fn.open("err.csv");
	arma::Mat <double> ERR;
	ERR.set_size(I, 3);
	ERR.fill(0);
	arma::Mat <double> x;	//input (hist sample)
	x.set_size(M, N);
	arma::Row <double> y;	//output (hist for sample x)
	y.set_size(N);
//
	arma::Cube <double> weights;	// weiths: input layer -> set of filters 
	weights.set_size(Q, kernel_size, kernel_size);
	arma::Row <double> bias;		// biases: input layer -> set of filters 
	bias.set_size(Q);
	arma::Cube <double> z;			// weighted average: input layer -> set of filters (input)
	z.set_size(Q, M, N);
	arma::Cube <double> a;			// neurons responce: from set of filters (outnput)
	a.set_size(Q, M, N);
	arma::Mat <int> sqmn;			// reference: long vector index for given q,m,n (filter number,filter row,filter column) 
	sqmn.set_size(V,3);
	arma::Row <double> v;			// all filters fed to vector 'v' (scan column by column)
	v.set_size(V);
	arma::Row <double> z_star;		// weighted average: hidden layer (filters) to output layer
	z_star.set_size(N);
	arma::Mat <double> weights_star;// weiths: long vector (fused filters) -> output layer 
	weights_star.set_size(V, N);
	arma::Row <double> bias_star;   // biases: long vector (fused filters) -> output layer  
	bias_star.set_size(N);
	arma::Row <double> a_star;		// neurons responce of output layer (final output)
	a_star.set_size(N);

	arma::Row <double> Dbias_star;	// derivatives by 'bias_star'
	Dbias_star.set_size(N);
	arma::Mat <double> Dweights_star;// derivatives by 'weights_star'
	Dweights_star.set_size(V, N);
	arma::Row <double> Dbias;		// derivatives by 'bias'
	Dbias.set_size(Q);
	arma::Cube <double> Dweights;	// derivatives by 'weights'
	Dweights.set_size(Q, kernel_size, kernel_size);
//
	// filling reference list
	for (s = 0; s < V; s++)
	{
		q = (int)(s / (Mf * Nf));
		n = (int)((s - q*Mf*Nf)/Mf);
		m = s - q * Mf*Nf - n * Mf;
		sqmn(s, 0) = q;
		sqmn(s, 1) = m;
		sqmn(s, 2) = n;
	}
//
	// random fill of all weights and biases:
	for (p = 0; p < N; p++)
	{
		bias_star(p) = arma::randn();
		for (s = 0; s < V; s++)
			weights_star(s, p) = arma::randn();
	}
	for (q = 0; q < Q; q++)
	{
		bias_star(q) = arma::randn();
		for (alpha = 0; alpha < kernel_size; alpha++)
			for (beta = 0; beta < kernel_size; beta++)
				weights(q, alpha, beta) = arma::randn();
	}
//
	//main iteration loop starts here:
	for (it = 0; it < I; it++)
	{
		Dbias_star.fill(0);
		Dweights_star.fill(0);
		Dbias.fill(0);
		Dweights.fill(0);
		// going over all samples:
		for (i = M; i < train_size - forcast_range; i++)
		{
			ERR(it, 0) = it;
			// forward propagation through neural network for a sample (i,i+M)
			// filling x(sample) and y(forcast)
			for (k = 0; k < N; k++)
			{
				y(k) = train_dataset(i + forcast_range, k);	//rates at date 'i+fwd_range' for tenor number 'k' (output)
				for(j = 0; j < M; j++ )
					x(j,k) = train_dataset(i - M + j, k);	//rates at date 'j' for tenor number 'k' (input)	
			}
			// weighted average z(qmn):
			for (q = 0; q < Q; q++)			// over all filters
			{
				for (m = 0; m < Mf; m++)	// by rows
				{
					for (n = 0; n < Nf; n++)// by columns
					{
						sum = 0;
						for (alpha = 0; alpha < kernel_size; alpha++)
							for (beta = 0; beta < kernel_size; beta++)
								sum = sum + x(m + alpha, n + beta) * weights(q,alpha, beta);
						z(q, m, n) = sum + bias(q);			//weighted average
						a(q, m, n) = sigmoid(z(q, m, n));	//neuroresponce
					}
				}
			}
			//filling vectorized_concatenated object ("long vector")
			for (s = 0; s < V; s++)
				v(s) = a(sqmn(s,0), sqmn(s, 1), sqmn(s, 2));	// a(q,m,n) for given s
//
			// z*: and a*:
			for (p = 0; p < N; p++)
			{
				sum = 0;
				for (s = 0; s < V; s++)
				{
					sum = sum + v(s) * weights_star(s, p);
				}
				z_star(p) = sum + bias_star(p);
				a_star(p) = sigmoid(z_star(p));	// output layer neuroresponce
				ERR(it,1) = ERR(it,1) + pow(a_star(p) - y(p), 2) / train_size;	//accumulation of errors
			}
//  
			// Calculation of derivatives 
			double tau_p; 
			for (p = 0; p < N; p++)
			{
				tau_p = (a_star(p) - y(p)) * sigmoid_d(z_star(p)) ;	
				Dbias_star(p) = Dbias_star(p) + tau_p / train_size;
				for(s = 0; s < V; s++)
					Dweights_star(s,p) = Dweights_star(s, p) + tau_p * v(s) / train_size;
			}
			for (q = 0; q < Q; q++)
			{
				sum_p = 0;
				for (p = 0; p < N; p++)
				{
					tau_p = (a_star(p) - y(p)) * sigmoid_d(z_star(p));
					sum_mn = 0;
					for(m=0;m<Mf;m++)
					{
						for(n=0;n<Nf;n++)
						{
							s = q * Mf*Nf + Mf * n + m;
							sum_mn = sum_mn + sigmoid_d(z(q, m, n)) * weights_star(s, p);
						}
					}
					sum_p = sum_p + tau_p * sum_mn;
				}
				Dbias(q) = Dbias(q) + sum_p / train_size;	
			}
			for (q = 0; q < Q; q++)
			{
				for (alpha = 0; alpha < kernel_size; alpha++)
				{
					for (beta = 0; beta < kernel_size; beta++)
					{
						sum_p = 0;
						for (p = 0; p < N; p++)
						{
							tau_p = (a_star(p) - y(p)) * sigmoid_d(z_star(p));
							sum_mn = 0;
							for (m = 0; m < Mf; m++)
							{
								for (n = 0; n < Nf; n++)
								{
									s = q * Mf*Nf + Mf * n + m;
									sum_mn = sum_mn + sigmoid_d(z(q, m, n)) * weights_star(s, p) * x(m+alpha,n+beta);
								}
							}
							sum_p = sum_p + tau_p * sum_mn;
						}
						Dweights(q, alpha, beta) = Dweights(q, alpha, beta) + sum_p / train_size;
					}
				}
			}
		}	//forward/back propagation & derivatives calculation for current iteration ends here
//
		//validation starts here:
		for (i = M; i < validation_size - forcast_range; i++)
		{
			// forward propagation through neural network
			// filling x(sample) and y(forcast)
			for (k = 0; k < N; k++)
			{
				y(k) = validation_dataset(i + forcast_range, k);	//rates at date 'i+fwd_range' for tenor number 'k' (output)
				for (j = 0; j < M; j++)
					x(j, k) = validation_dataset(i - M + j, k);				//rates at date 'i' for tenor number 'k' (input)	
			}
			// z(qmn):
			for (q = 0; q < Q; q++)
			{
				for (m = 0; m < Mf; m++)
				{
					for (n = 0; n < Nf; n++)
					{
						sum = 0;
						for (alpha = 0; alpha < kernel_size; alpha++)
							for (beta = 0; beta < kernel_size; beta++)
								sum = sum + x(m + alpha, n + beta) * weights(q, alpha, beta);
						z(q, m, n) = sum + bias(q);			//weighted average
						a(q, m, n) = sigmoid(z(q, m, n));	//output
					}
				}
			}
			//filling vectorized_concatenated object
			for (s = 0; s < V; s++)
				v(s) = a(sqmn(s, 0), sqmn(s, 1), sqmn(s, 2));	// a(q,m,n) for given s
			// z*: and a*:
			for (p = 0; p < N; p++)
			{
				sum = 0;
				for (s = 0; s < V; s++)
				{
					sum = sum + v(s) * weights_star(s, p);
				}
				z_star(p) = sum + bias_star(p);
				a_star(p) = sigmoid(z_star(p));
				ERR(it, 2) = ERR(it, 2) + pow(a_star(p) - y(p), 2) / validation_size;	//accumulation of errors
			}
		}
		if (minValErr > ERR(it, 2))
		{
			i_minValErr = it;
			minValErr = ERR(it, 2);
		}
		//gradient downhill step:
		for (p = 0; p < N; p++)
		{
			bias_star(p) = bias_star(p) - h * Dbias_star(p);
			for (s = 0; s < V; s++)
			{
				weights_star(s, p) = weights_star(s, p) - h * Dweights_star(s,p);
			}
		}
		for (q = 0; q < Q; q++)
		{
			for (alpha = 0; alpha < kernel_size; alpha++)
			{
				for (beta = 0; beta < kernel_size; beta++)
				{
					weights(q, alpha, beta) = weights(q, alpha, beta) - h * Dweights(q, alpha, beta);
				}
			}
		}
		cout << "|";
	}	//iteration loop ends here
//
	cout << endl << "Final training error= " << ERR(I - 1, 1);
	cout << endl << "Minimal validation error= " << minValErr << " at " << i_minValErr << " -th iteration";
	int it_min = 0;
	double min_val_err = 1e12;
	for (it = 1; it < I; it++)
		if (ERR(it, 2) < min_val_err)
		{
			it_min = it;
			min_val_err = ERR(it, 2);
		}
	cout << endl << "val err is min for it=" << it_min;
	cout << endl << it << " iterations completed" << endl;

	//ERR.save(out_obj_fn, arma::raw_ascii);
//	error vs iterations saving to a file:
	out_obj_fn << "iteration,err,validation_err";
	for (it = 0; it < I; it++)
		out_obj_fn << endl << ERR(it, 0) << "," << ERR(it, 1) << "," << ERR(it, 2);
	out_obj_fn.close();
	//



}
//	
int main(int argc, char **argv)
{
	cout << "Reading input files: ";
	read_input_files(argv[1]);
	iteration_loop();
	//
		// errors vs iteration: python code call
	system("python view.py");
	//
	return 0;
}
//	Copyright © 2019 Yashkir Consulting
