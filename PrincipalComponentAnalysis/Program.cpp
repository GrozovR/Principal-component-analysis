#include <iostream>
#include "Eigen/Dense"
#include "pca.h"
#include <fstream>

using Eigen::MatrixXd;
using namespace std;

int main()
{

	int rows = 150; //number of samples 
	int columns = 4; //number of components
	string pathData = "D:\\test_data.txt";

	MatrixXd m(rows, columns);

	ifstream istrm;
	istrm.open(pathData);

	double a{ 0 };
	for (int i = 0; i < 150; i++)
	{
		istrm >> a;
		m(i, 0) = a;
		istrm >> a;
		m(i, 1) = a;
		istrm >> a;
		m(i, 2) = a;
		istrm >> a;
		m(i, 3) = a;
	}
	istrm.close();

	MatrixXd uReduced = PCA::Compute(m);
	MatrixXd mReduced = m*uReduced;

	cout << mReduced << endl;
}