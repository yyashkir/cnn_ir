#pragma once

using namespace std;
#include <armadillo>

void log_record(string note);
double volatility(double tenor, double StoK_ratio, arma::Mat<double> vol_surf, int numb_t, int numb_ratios);
double fwdBS(int cp, double F, double K, double T, double sigma);
double cdf(double x);
double d1fwd(double F, double K, double T, double sigma);
double d2fwd(double d1, double T, double sigma);
double interpolation(arma::Mat<double> y0, int n, int rank, double t);
void plot1graph(string out_file, int col_x, int col_y, string xlabel, string ylabel, int show_graph, int logscale);
void plot2graphs(string out_file, int col_x, int col_y1, int col_y2, string xlabel, string ylabel, int show_graph);
void make_chart(int k, string out_file, int col_x, int col_y1, int col_y2, int col_y3, string xlabel, string ylabel, string title, int show_graph);
double max(double a, double b);
double mean_col(arma::Mat<double> M, int col);
double min_col(arma::Mat<double> M, int col);
double percentile_col(arma::Mat<double> U, double confidence, int col);
string date_now();
//	Copyright © 2018 Yashkir Consulting

string time_now(void);

void timestamp(void);
void nelmin(double fn(double x[]), int n, double start[], double xmin[],
	double *ynewlo, double reqmin, double step[], int konvge, int kcount,
	int *icount, int *numres, int *ifault, double objf[]);

double constraints_penalty(arma::Row<double> x, arma::Row<double> y, arma::Row<double> ksi,
	arma::Mat<double> range_A, arma::Mat<double> range_B,
	int nx, int ny, int nksi,
	int dual,
	double penalty_weight);

double sigmoid(double x);
double sigmoid_d(double x);
