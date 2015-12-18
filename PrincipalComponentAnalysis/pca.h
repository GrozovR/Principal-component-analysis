#ifndef PCA_H
#define PCA_H

#include <iostream>
#include <assert.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace Eigen;
using namespace std;

class PCA
{
public:

	/*
	 * Computes the princcipal component of a given matrix. Computation steps:
	 * Compute the mean image.
	 * Compute standard dispersion
	 * Normalizing input data
	 * Compute the covariance matrix
	 * Calculate the eigenvalues and eigenvectors for the covariance matrix

	 * @input MatrixXd D the data samples matrix.
	 *
	 * @returns MatrixXd The eigenVectors of principal component 
	 */

	static MatrixXd Compute(MatrixXd D)
	{
		// 1. Compute the mean image 
		MatrixXd mean(1, D.cols());
		mean.setZero();

		for (int i = 0; i < D.cols(); i++)
		{
			mean(0, i) = D.col(i).mean();
		}


		// 2. Compute standard dispersion
		MatrixXd s(1, D.cols());
		for (int i = 0; i < D.cols(); i++)
		{
			double ss{ 0 };

			for (int j = 0; j < D.rows(); j++)
			{
				double tmp = (D(j, i) - mean(0, i));
				tmp *= tmp;
				ss += tmp;
			}
			ss /= (D.rows() - 1);
			s(0, i) = sqrt(ss);
		}
		//cout << s << endl;


		// 3. Normalizing input data 
		MatrixXd normD(D);
		normD.setZero();
		for (int i = 0; i < D.cols(); i++)
		{
			for (int j = 0; j < D.rows(); j++)
			{
				normD(j, i) = (D(j, i) - mean(0, i)) / s(0, i);
			}
		}
		//cout << normD << endl;


		// 3. Compute the covariance matrix
		mean.setZero();

		for (int i = 0; i < normD.cols(); i++)
		{
			mean(0, i) = normD.col(i).mean();
		}

		MatrixXd covariance(D.cols(), D.cols());
		covariance.setZero();
		for (int k = 0; k < D.cols(); k++)
		{
			for (int j = 0; j < D.cols(); j++)
			{
				double sum{ 0 };
				for (int i = 0; i < D.rows(); i++)
				{
					sum += (normD(i, k) - mean(0, k))*(normD(i, j) - mean(0, j));
				}
				covariance(k, j) = sum / (D.rows() - 1);
			}
		}
		// cout << covariance << endl;


		// 4. Calculate the eigenvalues and eigenvectors for the covariance matrix
		EigenSolver<MatrixXd> solver(covariance);
		MatrixXd eigenVectors = solver.eigenvectors().real();
		VectorXd eigenValues = solver.eigenvalues().real();


		// 5. Find out an number of most important principal components with broken stick method   
		int newDimension = brockenStickMethod(eigenValues);

		// 6. Return the matrix of eigenVectors appropriate most important components
		MatrixXd featureVectors(D.cols(), newDimension);

		VectorXd sortEV(eigenValues);
		sort(sortEV.derived().data(), sortEV.derived().data() + sortEV.derived().size());

		for (int i = 0; i < newDimension; i++)
		{
			double max = sortEV(sortEV.rows() - 1 - i);

			for (int j = 0; j < eigenValues.size(); j++)
			{
				if (max == eigenValues(j))
				{
					VectorXd tmp = eigenVectors.col(j);
					featureVectors.col(i) = tmp;
				}
			}
		}
		
		return featureVectors;
	}

private:
	static int brockenStickMethod(VectorXd eigenValues)
	{
		// Compute number of components, which most influed
		double trc{ eigenValues.sum() };
		VectorXd l(eigenValues.size());
		int countPrincComp{ 0 };
		l.setZero();

		for (int i = 0; i < eigenValues.size(); i++)
		{
			double sum{ 0 };
			for (int j = i; j < eigenValues.size(); j++)
			{
				double coef = 1;
				sum += (coef / j);
			}
			l(i) = sum / eigenValues.size();
		}


		VectorXd sortEV(eigenValues);
		sort(sortEV.derived().data(), sortEV.derived().data() + sortEV.derived().size());

		for (int i = 0; i < sortEV.size(); i++)
		{
			if ((sortEV(i) / trc) >l(i)) { countPrincComp++; }
		}

		return countPrincComp;
	}
};

#endif // PCA_H