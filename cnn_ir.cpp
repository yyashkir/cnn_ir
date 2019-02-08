//	Copyright © 2018 Yashkir Consulting
/*
26/12/2018	-	hjm model fast calibration project started
				random driver generation moved out of Monte Carlo loop (cube ji6)
28/12/2018	-	end
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

arma::Row <int> tenors;		//tenors are expressed using  time point array  
arma::Mat <double> historical_dataset;		// input hist data rates for all tenor over period of time
string historical_datafile;	// file with historical yield rates 
arma::Mat <double> train_dataset;		// input hist data rates for all tenor over period of time
arma::Mat <double> validation_dataset;		// input hist data rates for all tenor over period of time

double pi = 3.14159265359;
int i_start_train;		// start day from historical yield data
int train_size;	// train size
int i_start_validation;			// length (days) of historical rates used for calibration
int validation_size;
int N;	// yield length
int M;	// sample size
int Mf;	// filter length
int Nf;	// filter width
int V;	// vectorize_concatenate length
int i_hist;			// total number of lines in file with historical yields
int	forcast_range;		// number of time steps forward for prediction
int kernel_size;
int Q; // number of filters
int  iteration_max_number;
double h;	// learning rate

clock_t start;

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
		>> s >> iteration_max_number
		>> s >> h
		;
	infile.close();
	//
		//reading historical rates 
	ifstream data_stream(historical_datafile.c_str());
	data_stream >> N;		// number of tenors (=columns)
	tenors.set_size(N);
	data_stream >> i_hist;		// number of lines in file with historical yields
	for (k = 0; k < N; k++)
		data_stream >> tenors(k);
	//
	historical_dataset.set_size(i_hist, N);
	for (i = 0; i < i_hist; i++)
		for (k = 0; k < N; k++)
			data_stream >> historical_dataset(i, k);
	data_stream.close();
	cout << "\nHistorical_dataset number of rows: " << historical_dataset.n_rows << endl;
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
	/*train_dataset.print(st.str());
	validation_dataset.print(sv.str());*/
	Mf = M - kernel_size + 1;
	Nf = N - kernel_size + 1;
	V = Q * Mf * Nf;
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
	ERR.set_size(iteration_max_number, 3);
	ERR.fill(0);
	arma::Mat <double> x;	//input (hist sample)
	x.set_size(M, N);
	arma::Row <double> y;	//output (hist)
	y.set_size(N);
//****************************************
	arma::Cube <double> weights;
	weights.set_size(Q, kernel_size, kernel_size);
	arma::Row <double> bias;
	bias.set_size(Q);
	arma::Cube <double> z;
	z.set_size(Q, M, N);
	arma::Cube <double> a;
	a.set_size(Q, M, N);
	arma::Mat <int> sqmn;
	sqmn.set_size(V,3);
	arma::Row <double> v;
	v.set_size(V);
	arma::Row <double> z_star;
	z_star.set_size(N);
	arma::Mat <double> weights_star;
	weights_star.set_size(V, N);
	arma::Row <double> bias_star;
	bias_star.set_size(N);
	arma::Row <double> a_star;
	a_star.set_size(N);

	arma::Row <double> Dbias_star;
	Dbias_star.set_size(N);
	arma::Mat <double> Dweights_star;
	Dweights_star.set_size(V, N);
	arma::Row <double> Dbias;
	Dbias.set_size(Q);
	arma::Cube <double> Dweights;
	Dweights.set_size(Q, kernel_size, kernel_size);

	//
	for (s = 0; s < V; s++)
	{
		q = (int)(s / (Mf * Nf));
		n = (int)((s - q*Mf*Nf)/Mf);
		m = s - q * Mf*Nf - n * Mf;
		sqmn(s, 0) = q;
		sqmn(s, 1) = m;
		sqmn(s, 2) = n;
	}
//	sqmn.print("sqmn");
	
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

	int flag = 0;
	//main iteration loop starts here:
	for (it = 0; it < iteration_max_number; it++)
	{
		Dbias_star.fill(0);
		Dweights_star.fill(0);
		Dbias.fill(0);
		Dweights.fill(0);
		//over all samples:
		for (i = M; i < train_size - forcast_range; i++)
		{
			ERR(it, 0) = it;
			// forward propagation through neural network
			// filling x(sample) and y(forcast)
			for (k = 0; k < N; k++)
			{
				y(k) = train_dataset(i + forcast_range, k);	//rates at date 'i+fwd_range' for tenor number 'k' (output)
				for(j = 0; j < M; j++ )
					x(j,k) = train_dataset(i - M + j, k);				//rates at date 'i' for tenor number 'k' (input)	
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
								sum = sum + x(m + alpha, n + beta) * weights(q,alpha, beta);
						z(q, m, n) = sum + bias(q);			//weighted average
						a(q, m, n) = sigmoid(z(q, m, n));	//output
					}
				}
			}
			//filling vectorized_concatenated object
			for (s = 0; s < V; s++)
				v(s) = a(sqmn(s,0), sqmn(s, 1), sqmn(s, 2));	// a(q,m,n) for given s

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
				ERR(it,1) = ERR(it,1) + pow(a_star(p) - y(p), 2) / train_size;	//accumulation of errors
			}
  
			//derivatives calculation
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
			

		}	//single iteration forward/back propagation & derivatives calculation ends here

		//validation starts here:
		//for (i = 0; i < validation_size - fwd_range; i++)
		//{
		//	// forward propagation through neural network
		//	for (k = 0; k < N; k++)
		//	{
		//		x(k) = validation_dataset(i, k);				//input
		//		y(k) = validation_dataset(i + fwd_range, k);	//output
		//	}
		//	z_01 = x * weight_01 + bias_01;		// weighted average for entry to hidden layer '1'
		//	for (k = 0; k < N; k++)
		//	{
		//		a_1(k) = sigmoid(z_01(k));		//out of hidden layer '1'
		//	}
		//	z_12 = a_1 * weight_12 + bias_12;	// weighted average for entry to output layer '2'
		//	for (k = 0; k < N; k++)
		//	{
		//		a_2(k) = sigmoid(z_12(k));		// out of output layer '2'
		//		ERR(it, 2) = ERR(it, 2) + pow(a_2(k) - y(k), 2) / validation_size;	//error accumulation
		//	}

		//}	//validation end
		//if (it > 1 && ERR(it - 2, 2) >= ERR(it - 1, 2) && ERR(it - 1, 2) <= ERR(it, 2) || it == iteration_max_number - 1)
		//{
		//	cout << endl << it - 1 << " " << ERR(it - 1, 2) << " (minimal validation error)";
		//	//illustration
		//	i = 0;
		//	// prediction example:
		//	for (k = 0; k < N; k++)
		//	{
		//		x(k) = validation_dataset(i, k);				//input
		//		y(k) = validation_dataset(i + fwd_range, k);	//output
		//	}
		//	z_01 = x * weight_01 + bias_01;		// weighted average for entry to hidden layer '1'
		//	for (k = 0; k < N; k++)
		//	{
		//		a_1(k) = sigmoid(z_01(k));		//out of hidden layer '1'
		//	}
		//	z_12 = a_1 * weight_12 + bias_12;	// weighted average for entry to output layer '2'
		//	for (k = 0; k < N; k++)
		//	{
		//		a_2(k) = sigmoid(z_12(k));		// out of output layer '2'
		//	}
		//	y.print("\ny for validation set i=0");
		//	a_2.print("a_2 modelled");
		//	cout << endl << "optimal weights and biases:";
		//	weight_01.print("\nw01");	weight_01.save("weight_01.opt", arma::raw_ascii);
		//	bias_01.print("b01");		bias_01.save("bias_01.opt", arma::raw_ascii);
		//	weight_12.print("w12");		weight_12.save("weight_12.opt", arma::raw_ascii);
		//	bias_12.print("b12");		bias_12.save("bias_12.opt", arma::raw_ascii);
		//}
		////
		//cout << '.';
		////
		//		// derivatives by weights calculation:
		//for (i = back_range; i < train_size - fwd_range; i++)
		//{
		//	for (k = 0; k < N; k++)
		//	{
		//		for (j = 0; j < N; j++)
		//		{
		//			w_d01(k, j) = w_d01(k, j) + delta_01(i, j) * a_1all(i, k) / train_size;
		//			w_d12(k, j) = w_d12(k, j) + delta_12(i, j) * a_2all(i, k) / train_size;
		//		}
		//	}
		//}
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
	}	//iteration loop end

	int it_min = 0;
	double min_val_err = 1e12;
	for (it = 1; it < iteration_max_number; it++)
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
	for (it = 0; it < iteration_max_number; it++)
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
