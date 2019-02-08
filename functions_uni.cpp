//	Copyright © 2018 Yashkir Consulting
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <time.h>
#include "functions_uni.h"
#include <armadillo>
# include <cstdlib>
# include <iomanip>
# include <ctime>
# include <cmath>
using namespace std;
//
double interpolation(arma::Mat<double> y0, int n, int rank, double t)
{
	// interpolation for value 'y' for tenor t using data y0
	int p;
	double y;
	for (p = 1; p < n; p++)
	{
		if (t >= y0(p-1,0) && t <= y0(p,0))	break;
	}
	if (t <= y0(0,0) ) p = 0;
	if (t >= y0(n-1,0))  p = n - 2;
	y = y0(p, rank) + (y0(p + 1,rank) - y0(p,rank)) * (t - y0(p,0)) / (y0(p + 1,0) - y0(p,0));
	return y;
}
//
void plot1graph(string out_file, int col_x, int col_y, string xlabel, string ylabel, int show_graph, int logscale)
{
	stringstream plot_string, show_string;
	string plot;
	string scale = "";
	if (logscale == 1) scale = "set logscale y 2;";
	plot_string << "set terminal png;"
		<< endl << "set output '" << xlabel<< ".png';"
		<< endl << "set grid;"
		<< endl << "set grid mytics;"
		<< endl << scale
		<< endl << "set xlabel '" << xlabel << "';"
		<< endl << "set ylabel '" << ylabel << "';"
		<< endl << "plot '" << out_file << "' using " << col_x << ":" << col_y << " with points  pt 7 ps 0.5;";
	plot = plot_string.str();
	ofstream plot_command_file;
	plot_command_file.open("plotcommand");
	plot_command_file << plot;
	plot_command_file.close();
	int ret = system("wgnuplot plotcommand");
	if (ret == 0)
	{
		if (show_graph > 1)
		{
			show_string << "mspaint " << xlabel << ".png";
			string tmp = show_string.str();
			const char *show_graph = tmp.c_str();
			system(show_graph);
		}
	}
	else
	{
		cout<<"gnuplot failure\n";
	}
}
//
void plot2graphs(string out_file, int col_x, int col_y1, int col_y2, string xlabel,string ylabel, int show_graph)
{
	stringstream plot_string, show_string;
	string plot;
	plot_string << "set terminal png;"
		<< endl << "set output " << "'out_" <<col_x<<col_y1<<col_y2 << ".png';"
		<< endl << "set grid;"
		<< endl << "set key autotitle columnhead;"
		<< endl << "set xlabel '"<<xlabel<<"';"
		<< endl << "set ylabel '"<< ylabel << "';"
		<< endl << "plot '" << out_file << "' using "<< col_x <<":" << col_y1 << " with lines, '" 
							<< out_file << "' using "<< col_x <<":" << col_y2 << " with lines;";
	plot = plot_string.str();
	ofstream plot_command_file;
	plot_command_file.open("plotcommand");
	plot_command_file << plot;
	plot_command_file.close();
	int ret = system("wgnuplot plotcommand");
	if (ret == 0)
	{
		if (show_graph != 0 )
		{
			show_string << "mspaint " << "out_" << col_x << col_y1 << col_y2 << ".png";
			string tmp = show_string.str();
			const char *show_graph = tmp.c_str();
			system(show_graph);
		}
	}
	else
	{
		cout<<"gnuplot failure\n";
	}
}
//
void make_chart(int k, string out_file, int col_x, int col_y1, int col_y2, int col_y3
			   ,string xlabel, string ylabel, string title, int show_graph)
{
	stringstream plot_string, show_string;
	string plot;
	title = title.substr(0, 3)  + "_" + to_string(k);
	plot_string << "set terminal png noenhanced;"
		<< endl << "set output '" <<title <<  ".png';"
		<< endl << "set grid;"
		<< endl << "set key autotitle columnhead;"
		<< endl << "set title '" << title  <<"';"
		<< endl << "set xlabel '" << xlabel << "';"
		<< endl << "set ylabel '" << ylabel << "';"
		<< endl << "plot '" << out_file << "' using " << col_x << ":" << col_y1 << " with points pt 7 ps 0.5, '" 
							<< out_file << "' using " << col_x << ":" << col_y2 << " with lines, '"
							<< out_file << "' using " << col_x << ":" << col_y3 << " with lines; "
						;
	plot = plot_string.str();
	ofstream plot_command_file;
	plot_command_file.open("plotcommand");
	plot_command_file << plot;
	plot_command_file.close();
	int ret = system("wgnuplot plotcommand");
	if (ret == 0)
	{
		if (show_graph > 1)
		{
			show_string << "mspaint " << title<< ".png";
			string tmp = show_string.str();
			const char *show_graph = tmp.c_str();
			system(show_graph);
		}
	}
	else
	{
		cout<<"gnuplot failure\n";
	}
}
//
void log_record(string note)
{
	ofstream f;
	f.open("log_file.txt", ios::app);
	f << note;
	f.close();
}
//
string date_now()
{
	time_t _tm = time(NULL);
	struct tm *curtime = localtime(&_tm);
	string s = "\n";
	s.append(asctime(curtime));
	return(s);
}
//
double mean_col(arma::Mat<double> M, int col)
{
	int k;
	double sum = 0;
	int jmax = (int)M.n_rows;
	if (jmax < 0)
	{
		cout<<"ERROR! array for mean() is empty";
		exit(0);
	}
	for (k = 0; k < jmax; k++) sum = sum + M(k,col);
	return sum / jmax;
}
//
double min_col(arma::Mat<double> M, int col)
{
	int k;
	double v = 1e10;
	int jmax = (int)M.n_rows;
	if (jmax < 0)
	{
		cout<<"ERROR! array is empty";
		exit(0);
	}
	for (k = 0; k < jmax; k++) if(M(k, col) < v) v = M(k, col);
	return v;
}
//
double percentile_col(arma::Mat<double> U, double confidence, int col)
{
	int j;
	double w, dcp;
	int jmax = (int)U.n_rows;
	if (jmax == 1)
		return U(0,col);
	dcp = 1.0 / jmax;
	arma::Row<double> column;
	column.set_size(jmax);
	for (j = 0; j < jmax; j++) column(j) = U(j, col);
	column = sort(column);
	int j1 = (int)(confidence * jmax);
	if (j1 >= jmax) j1 = jmax - 1;
	w = column(j1) + (column(j1 + 1) - column(j1)) * (confidence - (1.0 * j1) / jmax) / dcp;
	return w;
}
//
double max(double a, double b)
{
	if (a >= b)return a;
	else return b;
}
//
double cdf(double x)
{
	// cumulative distribution function
	double	A1 = 0.31938153,
		A2 = -0.356563782,
		A3 = 1.781477937,
		A4 = -1.821255978,
		A5 = 1.330274429,
		RSQRT2PI = 0.39894228040143267793994605993438;
	double  K = 1.0 / (1.0 + 0.2316419 * fabs(x));
	double  cnd = RSQRT2PI * exp(-0.5 * x * x) *
		(K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
	if (x > 0)
		cnd = 1.0 - cnd;
	return cnd;
}
//
double d1fwd(double F, double K, double T, double sigma)
{
	return  (log(F / K) + 0.5*sigma*sigma*T) / (sigma * sqrt(T));
}
//
double d2fwd(double d1, double T, double sigma)
{
	return  d1 - sigma * sqrt(T);
}
//
double fwdBS(int cp, double F, double K, double T, double sigma)
{	// cp=1 for call, cp=-1 for put
	//F is forward price, K is strike, T is maturity, sigma is volatility
	if (F <= 0 && cp > 0) return 0;
	if (F <= 0 && cp < 0) return K;
	double d1 = d1fwd(F, K, T, sigma);
	double d2 = d2fwd(d1, T, sigma);
	return cp * (F * cdf(cp*d1) - K * cdf(cp*d2) );
}
//
double volatility(double tenor, double StoK_ratio, arma::Mat<double> vol_surf, int numb_t, int numb_ratios)
{
	// volatility for S/K ratio and tenor, using interpolation
	double vol = 0, x, y, x1 = 0, y1, x2 = 0, y2; // tenor y , ratio S/K  x
	double vol11, vol12, vol21, vol22, R1, R2;
	int i1 = 0, i2 = 0, j1, j2;
	x = StoK_ratio;
	y = tenor;
	/*
	x1,y2		x2,y2
	x,y
	x1,y1		x2,y2
	*/
	int j, i;
	for (i = 1; i < numb_ratios; i++)
	{
		if (x >= vol_surf(0,i) && x <= vol_surf(0,i + 1))
		{
			x1 = vol_surf(0, i);
			x2 = vol_surf(0, i+1);
			i1 = i;
			i2 = i + 1;
		}
	}
	if (x < vol_surf(0, 1))
	{
		x1 = vol_surf(0, 1);
		x2 = vol_surf(0, 2);
		i1 = 1;
		i2 = 2;
	}
	if (x > vol_surf(0,numb_ratios))
	{
		x1 = vol_surf(0, numb_ratios-1);
		x2 = vol_surf(0, numb_ratios);
		i1 = numb_ratios - 1;
		i2 = numb_ratios;
	}
	for (j = 1; j < numb_t; j++)
	{
		if (y >= vol_surf(j,0) && y <= vol_surf(j+1, 0))
		{
			y1 = vol_surf(j, 0);
			y2 = vol_surf(j+1, 0);
			j1 = j;
			j2 = j + 1;
		}
	}
	if (y < vol_surf(1, 0))
	{
		y1 = vol_surf(1, 0);
		y2 = vol_surf(2, 0);
		j1 = 1;
		j2 = 2;
	}
	if (y > vol_surf(numb_t,0))
	{
		y1 = vol_surf(numb_t-1, 0);
		y2 = vol_surf(numb_t, 0);
		j1 = numb_t - 1;
		j2 = numb_t;
	}
	vol11 = vol_surf(j1,i1);
	vol12 = vol_surf(j1, i2);
	vol21 = vol_surf(j2, i1);
	vol22 = vol_surf(j2, i2);
	R1 = ((x2 - x) / (x2 - x1)) * vol11 + ((x - x1) / (x2 - x1)) * vol21;
	R2 = ((x2 - x) / (x2 - x1)) * vol12 + ((x - x1) / (x2 - x1)) * vol22;
	vol = ((y2 - y) / (y2 - y1)) * R1 + ((y - y1) / (y2 - y1)) * R2;
	return vol;
}
//	Copyright © 2018 Yashkir Consulting
void timestamp(void)
//  Purpose:
//  TIMESTAMP prints the current YMDHMS date as a time stamp.
//	Licensing:
//	This code is distributed under the GNU LGPL license. 
//	Modified: 24 September 2003
//  Author: John Burkardt
{
# define TIME_SIZE 40
	static char time_buffer[TIME_SIZE];
	const struct tm *tm;
	size_t len;
	time_t now;
	now = time(NULL);
	tm = localtime(&now);
	len = strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);
	cout << time_buffer << "\n";
	return;
# undef TIME_SIZE
}
//
string time_now(void)
{
# define TIME_SIZE 40
	static char time_buffer[TIME_SIZE];
	const struct tm *tm;
	size_t len;
	time_t now;
	now = time(NULL);
	tm = localtime(&now);
	len = strftime(time_buffer, TIME_SIZE, "%d_%B_%Y_%I_%M_%S_%p", tm);
	return time_buffer;
# undef TIME_SIZE
}
//
void nelmin(double fn(double x[]), int n, double start[], double xmin[],
	double *ynewlo, double reqmin, double step[], int konvge, int kcount,
	int *icount, int *numres, int *ifault, double objf[])
	//****************************************************************************80
	//  Purpose:
	//    NELMIN minimizes a function using the Nelder-Mead algorithm.
	//  Discussion:
	//    This routine seeks the minimum value of a user-specified function.
	//    Simplex function minimisation procedure due to Nelder+Mead(1965),
	//    as implemented by O'Neill(1971, Appl.Statist. 20, 338-45), with
	//    subsequent comments by Chambers+Ertel(1974, 23, 250-1), Benyon(1976,
	//    25, 97) and Hill(1978, 27, 380-2)
	//    The function to be minimized must be defined by a function of
	//    the form
	//      function fn ( x, f )
	//      double fn
	//      double x(*)
	//    and the name of this subroutine must be declared EXTERNAL in the
	//    calling routine and passed as the argument FN.
	//    This routine does not include a termination test using the
	//    fitting of a quadratic surface.
	//  Licensing:
	//    This code is distributed under the GNU LGPL license. 
	//  Modified:
	//    27 February 2008
	//  Author:
	//    Original FORTRAN77 version by R ONeill.
	//    C++ version by John Burkardt.
	//  Reference:
	//    John Nelder, Roger Mead,
	//    A simplex method for function minimization,
	//    Computer Journal,
	//    Volume 7, 1965, pages 308-313.
	//    R ONeill,
	//    Algorithm AS 47:
	//    Function Minimization Using a Simplex Procedure,
	//    Applied Statistics,
	//    Volume 20, Number 3, 1971, pages 338-345.
	//  Parameters:
	//    Input, double FN ( double x[] ), the name of the routine which evaluates
	//    the function to be minimized.
	//    Input, int N, the number of variables.
	//    Input/output, double START[N].  On input, a starting point
	//    for the iteration.  On output, this data may have been overwritten.
	//    Output, double XMIN[N], the coordinates of the point which
	//    is estimated to minimize the function.
	//    Output, double YNEWLO, the minimum value of the function.
	//    Input, double REQMIN, the terminating limit for the variance
	//    of function values.
	//    Input, double STEP[N], determines the size and shape of the
	//    initial simplex.  The relative magnitudes of its elements should reflect
	//    the units of the variables.
	//    Input, int KONVGE, the convergence check is carried out 
	//    every KONVGE iterations.
	//    Input, int KCOUNT, the maximum number of function 
	//    evaluations.
	//    Output, int *ICOUNT, the number of function evaluations 
	//    used.
	//    Output, int *NUMRES, the number of restarts.
	//    Output, int *IFAULT, error indicator.
	//    0, no errors detected.
	//    1, REQMIN, N, or KONVGE has an illegal value.
	//    2, iteration terminated because KCOUNT was exceeded without convergence.
{
	double ccoeff = 0.5;
	double del;
	double dn;
	double dnn;
	double ecoeff = 2.0;
	double eps = 0.001;
	int i;
	int ihi;
	int ilo;
	int j;
	int jcount;
	int l;
	int nn;
	double *p;
	double *p2star;
	double *pbar;
	double *pstar;
	double rcoeff = 1.0;
	double rq;
	double x;
	double *y;
	double y2star;
	double ylo;
	double ystar;
	double z;
//
	int mycount = 0;
	//
	//  Check the input parameters.
	if (reqmin <= 0.0)
	{
		*ifault = 1;
		return;
	}
	if (n < 1)
	{
		*ifault = 1;
		return;
	}
	if (konvge < 1)
	{
		*ifault = 1;
		return;
	}
	p = new double[n*(n + 1)];
	pstar = new double[n];
	p2star = new double[n];
	pbar = new double[n];
	y = new double[n + 1];
	*icount = 0;
	*numres = 0;
	jcount = konvge;
	dn = (double)(n);
	nn = n + 1;
	dnn = (double)(nn);
	del = 1.0;
	rq = reqmin * dn;
	//
	//  Initial or restarted loop.
	//
	for (; ; )
	{
		for (i = 0; i < n; i++)
		{
			p[i + n * n] = start[i];
		}
		y[n] = fn(start);
		*icount = *icount + 1;
		for (j = 0; j < n; j++)
		{
			x = start[j];
			start[j] = start[j] + step[j] * del;
			for (i = 0; i < n; i++)
			{
				p[i + j * n] = start[i];
			}
			y[j] = fn(start);
			*icount = *icount + 1;
			start[j] = x;
		}
		//                    
		//  The simplex construction is complete.
		//                    
		//  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
		//  the vertex of the simplex to be replaced.
		//                
		ylo = y[0];
		ilo = 0;
		for (i = 1; i < nn; i++)
		{
			if (y[i] < ylo)
			{
				ylo = y[i];
				ilo = i;
			}
		}
		//
		//  Inner loop.
		//
		for (; ; )
		{
			if (kcount <= *icount)
			{
				break;
			}
			*ynewlo = y[0];
			ihi = 0;

			for (i = 1; i < nn; i++)
			{
				if (*ynewlo < y[i])
				{
					*ynewlo = y[i];
					ihi = i;
				}
			}
			cout <<endl << *icount <<"  "<< *ynewlo;
			objf[mycount] = *ynewlo;
			mycount++;
			//
			//  Calculate PBAR, the centroid of the simplex vertices
			//  excepting the vertex with Y value YNEWLO.
			//
			for (i = 0; i < n; i++)
			{
				z = 0.0;
				for (j = 0; j < nn; j++)
				{
					z = z + p[i + j * n];
				}
				z = z - p[i + ihi * n];
				pbar[i] = z / dn;
			}
			//
			//  Reflection through the centroid.
			//
			for (i = 0; i < n; i++)
			{
				pstar[i] = pbar[i] + rcoeff * (pbar[i] - p[i + ihi * n]);
			}
			ystar = fn(pstar);
			*icount = *icount + 1;
			//
			//  Successful reflection, so extension.
			//
			if (ystar < ylo)
			{
				for (i = 0; i < n; i++)
				{
					p2star[i] = pbar[i] + ecoeff * (pstar[i] - pbar[i]);
				}
				y2star = fn(p2star);
				*icount = *icount + 1;
				//
				//  Check extension.
				//
				if (ystar < y2star)
				{
					for (i = 0; i < n; i++)
					{
						p[i + ihi * n] = pstar[i];
					}
					y[ihi] = ystar;
				}
				//
				//  Retain extension or contraction.
				//
				else
				{
					for (i = 0; i < n; i++)
					{
						p[i + ihi * n] = p2star[i];
					}
					y[ihi] = y2star;
				}
			}
			//
			//  No extension.
			//
			else
			{
				l = 0;
				for (i = 0; i < nn; i++)
				{
					if (ystar < y[i])
					{
						l = l + 1;
					}
				}

				if (1 < l)
				{
					for (i = 0; i < n; i++)
					{
						p[i + ihi * n] = pstar[i];
					}
					y[ihi] = ystar;
				}
				//
				//  Contraction on the Y(IHI) side of the centroid.
				//
				else if (l == 0)
				{
					for (i = 0; i < n; i++)
					{
						p2star[i] = pbar[i] + ccoeff * (p[i + ihi * n] - pbar[i]);
					}
					y2star = fn(p2star);
					*icount = *icount + 1;
					//
					//  Contract the whole simplex.
					//
					if (y[ihi] < y2star)
					{
						for (j = 0; j < nn; j++)
						{
							for (i = 0; i < n; i++)
							{
								p[i + j * n] = (p[i + j * n] + p[i + ilo * n]) * 0.5;
								xmin[i] = p[i + j * n];
							}
							y[j] = fn(xmin);
							*icount = *icount + 1;
						}
						ylo = y[0];
						ilo = 0;

						for (i = 1; i < nn; i++)
						{
							if (y[i] < ylo)
							{
								ylo = y[i];
								ilo = i;
							}
						}
						continue;
					}
					//
					//  Retain contraction.
					//
					else
					{
						for (i = 0; i < n; i++)
						{
							p[i + ihi * n] = p2star[i];
						}
						y[ihi] = y2star;
					}
				}
				//
				//  Contraction on the reflection side of the centroid.
				//
				else if (l == 1)
				{
					for (i = 0; i < n; i++)
					{
						p2star[i] = pbar[i] + ccoeff * (pstar[i] - pbar[i]);
					}
					y2star = fn(p2star);
					*icount = *icount + 1;
					//
					//  Retain reflection?
					//
					if (y2star <= ystar)
					{
						for (i = 0; i < n; i++)
						{
							p[i + ihi * n] = p2star[i];
						}
						y[ihi] = y2star;
					}
					else
					{
						for (i = 0; i < n; i++)
						{
							p[i + ihi * n] = pstar[i];
						}
						y[ihi] = ystar;
					}
				}
			}
			//
			//  Check if YLO improved.
			//
			if (y[ihi] < ylo)
			{
				ylo = y[ihi];
				ilo = ihi;
			}
			jcount = jcount - 1;

			if (0 < jcount)
			{
				continue;
			}
			//
			//  Check to see if minimum reached.
			//
			if (*icount <= kcount)
			{
				jcount = konvge;
				z = 0.0;
				for (i = 0; i < nn; i++)
				{
					z = z + y[i];
				}
				x = z / dnn;
				z = 0.0;
				for (i = 0; i < nn; i++)
				{
					z = z + pow(y[i] - x, 2);
				}
				if (z <= rq)
				{
					cout << endl << "z=" << z << "  rq=" << rq;
					break;
				}
			}
		}
		//
		//  Factorial tests to check that YNEWLO is a local minimum.
		//
		for (i = 0; i < n; i++)
		{
			xmin[i] = p[i + ilo * n];
		}
		*ynewlo = y[ilo];

		if (kcount < *icount)
		{
			*ifault = 2;
			break;
		}
		*ifault = 0;
		for (i = 0; i < n; i++)
		{
			del = step[i] * eps;
			xmin[i] = xmin[i] + del;
			z = fn(xmin);
			*icount = *icount + 1;
			if (z < *ynewlo)
			{
				*ifault = 2;
				break;
			}
			xmin[i] = xmin[i] - del - del;
			z = fn(xmin);
			*icount = *icount + 1;
			if (z < *ynewlo)
			{
				*ifault = 2;
				break;
			}
			xmin[i] = xmin[i] + del;
		}
		if (*ifault == 0)
		{
			break;
		}
		//
		//  Restart the procedure.
		//
		for (i = 0; i < n; i++)
		{
			start[i] = xmin[i];
		}
		del = eps;
		*numres = *numres + 1;
	}
	delete[] p;
	delete[] pstar;
	delete[] p2star;
	delete[] pbar;
	delete[] y;
	return ;
}
//****************************************************************************80
double sigmoid(double x) 
{
	return 1. / (1. + exp(-x));
}
//sigmoid function derivative:
double sigmoid_d(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}
