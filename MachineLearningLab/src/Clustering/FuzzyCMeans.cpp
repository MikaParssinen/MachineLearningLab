#include "FuzzyCMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
#include <numeric>
using namespace System::Windows::Forms; // For MessageBox


///  FuzzyCMeans class implementation  ///


// FuzzyCMeans function: Constructor for FuzzyCMeans class.//
FuzzyCMeans::FuzzyCMeans(int numClusters, int maxIterations, double fuzziness)
	: numClusters_(numClusters), maxIterations_(maxIterations), fuzziness_(fuzziness) {}


// fit function: Performs Fuzzy C-Means clustering on the given dataset and return the centroids of the clusters.//
void FuzzyCMeans::fit(const std::vector<std::vector<double>>& data) {
	// Create a copy of the data to preserve the original dataset
	std::vector<std::vector<double>> normalizedData = data;

	/* Implement the following:
		--- Initialize centroids randomly
		--- Initialize the membership matrix with the number of data points
		--- Perform Fuzzy C-means clustering
	*/

	// Initialize centroids randomly
	std::random_device rd;  // Seed for the random number generator
	std::mt19937 gen(rd());  // Mersenne Twister random number engine
	std::uniform_int_distribution<int> distribution(0, data.size() - 1);

	centroids_.clear();  // Clear any existing centroids

	// Choose random data points as initial centroids
	for (size_t i = 0; i < numClusters_; ++i) {
		size_t randomIndex = distribution(gen);  // Generate a random index
		centroids_.push_back(normalizedData[randomIndex]);  // Use the data point as a centroid
	}

	// Initialize the membership matrix with the number of data points
	initializeMembershipMatrix(data.size());

	bool centroidsChanged = true;
	int iter = 0;

	while (centroidsChanged && iter < maxIterations_) {
		std::vector<std::vector<double>> oldCentroids = centroids_;

		// Perform Fuzzy C-means clustering
		updateMembershipMatrix(normalizedData, centroids_);

		// Update centroids based on fuzzy memberships
		updateCentroids(normalizedData);

		// Check for convergence
		centroidsChanged = !areCentroidsEqual(oldCentroids, centroids_);
		iter++;
	}
	
}


// initializeMembershipMatrix function: Initializes the membership matrix with random values that sum up to 1 for each data point.//
void FuzzyCMeans::initializeMembershipMatrix(int numDataPoints) {
	membershipMatrix_.clear();
	membershipMatrix_.resize(numDataPoints, std::vector<double>(numClusters_, 0.0));


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	for (int i = 0; i < numDataPoints; ++i) {
		double sum = 0.0;
		for (size_t j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] = distribution(gen);
			sum += membershipMatrix_[i][j];
		}

		// Normalize membership values to sum up to 1 for each data point
		for (size_t j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] /= sum;
		}
	}




	/* Implement the following:
		--- Initialize membership matrix with random values that sum up to 1 for each data point
		---	Normalize membership values to sum up to 1 for each data point
	*/
	
	// TODO
}


// updateMembershipMatrix function: Updates the membership matrix using the fuzzy c-means algorithm.//
void FuzzyCMeans::updateMembershipMatrix(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>> centroids_) {
	
	for (size_t i = 0; i < data.size(); ++i) 
	{
		for (size_t j = 0; j < numClusters_; ++j) 
		{
			double distanceToCurrent = SimilarityFunctions::euclideanDistance(data[i], centroids_[j]);
			double sum = 0.0;

			for (size_t k = 0; k < numClusters_; ++k) 
			{
				double distanceToK = SimilarityFunctions::euclideanDistance(data[i], centroids_[k]);
				sum += std::pow(distanceToCurrent / distanceToK, 2.0 / (fuzziness_ - 1.0));
			}

			// Update the membership matrix with the new value
			membershipMatrix_[i][j] = 1.0 / sum;
		}

		// Normalize membership values to sum up to 1 for each data point
		double sumMembership = std::accumulate(membershipMatrix_[i].begin(), membershipMatrix_[i].end(), 0.0);
		for (size_t j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] /= sumMembership;
		}
	}
	/* Implement the following:
		---	Iterate through each data point
		--- Calculate the distance between the data point and the centroid
		--- Update the membership matrix with the new value
		--- Normalize membership values to sum up to 1 for each data point
	*/
	
	// TODO
	
}


// updateCentroids function: Updates the centroids of the Fuzzy C-Means algorithm.//
std::vector<std::vector<double>> FuzzyCMeans::updateCentroids(const std::vector<std::vector<double>>& data) {
	
	std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(data[0].size(), 0.0));

	for (size_t j = 0; j < numClusters_; ++j) {
		for (size_t d = 0; d < data[0].size(); ++d) {
			double numerator = 0.0;
			double denominator = 0.0;

			for (size_t i = 0; i < data.size(); ++i) {
				double membershipPow = std::pow(membershipMatrix_[i][j], fuzziness_);
				numerator += membershipPow * data[i][d];
				denominator += membershipPow;
			}

			newCentroids[j][d] = numerator / denominator;
		}
	}
	
	/* Implement the following:
		--- Iterate through each cluster
		--- Iterate through each data point
		--- Calculate the membership of the data point to the cluster raised to the fuzziness
	*/
	
	// TODO


	return centroids_; // Return the centroids
}


// predict function: Predicts the cluster labels for the given data points using the Fuzzy C-Means algorithm.//
std::vector<int> FuzzyCMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels; // Create a vector to store the labels
	labels.reserve(data.size()); // Reserve space for the labels

	for (const auto& point : data) {
		int closestCentroidIndex = -1; // Index of the closest centroid
		double maxMembership = -1.0;

		for (size_t i = 0; i < numClusters_; ++i) {
			double membership = calculateMembership(point, centroids_[i]);
			if (membership > maxMembership) {
				maxMembership = membership;
				closestCentroidIndex = static_cast<int>(i);
			}
		}

		// Add the label of the closest centroid to the labels vector
		labels.push_back(closestCentroidIndex);
	}

	return labels; // Return the labels vector
}

	/* Implement the following:
		--- Iterate through each point in the data
		--- Iterate through each centroid
		--- Calculate the distance between the point and the centroid
		--- Calculate the membership of the point to the centroid
		--- Add the label of the closest centroid to the labels vector
	*/
	
	//TODO


// calculateMembership function: Calculates the membership of a point to a centroid.//

double FuzzyCMeans::calculateMembership(const std::vector<double>& point, const std::vector<double>& centroid) const {
	double distanceToCurrent = SimilarityFunctions::euclideanDistance(point, centroid);
	double sum = 0.0;

	for (size_t k = 0; k < numClusters_; ++k) {
		double distanceToK = SimilarityFunctions::euclideanDistance(point, centroids_[k]);
		sum += std::pow(distanceToCurrent / distanceToK, 2.0 / (fuzziness_ - 1.0));
	}

	// Calculate the membership of the point to the centroid
	return 1.0 / sum;

}


bool FuzzyCMeans::areCentroidsEqual(const std::vector<std::vector<double>>& centroids1, const std::vector<std::vector<double>>& centroids2) const {
	if (centroids1.size() != centroids2.size())
		return false;

	for (size_t i = 0; i < centroids1.size(); ++i) {
		if (centroids1[i] != centroids2[i])
			return false;
	}

	return true;
}



/// runFuzzyCMeans: this function runs the Fuzzy C-Means clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
FuzzyCMeans::runFuzzyCMeans(const std::string& filePath) {
	DataPreprocessor DataPreprocessor;
	try {
		// Check if the file path is empty
		if (filePath.empty()) {
			MessageBox::Show("Please browse and select the dataset file from your PC.");
			return {}; // Return an empty vector since there's no valid file path
		}

		// Attempt to open the file
		std::ifstream file(filePath);
		if (!file.is_open()) {
			MessageBox::Show("Failed to open the dataset file");
			return {}; // Return an empty vector since file couldn't be opened
		}

		std::vector<std::vector<double>> dataset; // Create an empty dataset vector
		DataLoader::loadAndPreprocessDataset(filePath, dataset);

		// Use the all dataset for training and testing sets.
		double trainRatio = 1.0;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData);

		// Make predictions on the training data
		std::vector<int> labels = predict(trainData);

		// Calculate evaluation metrics
		// Calculate Davies BouldinIndex using the actual features and predicted cluster labels
		double daviesBouldinIndex = Metrics::calculateDaviesBouldinIndex(trainData, labels);

		// Calculate Silhouette Score using the actual features and predicted cluster labels
		double silhouetteScore = Metrics::calculateSilhouetteScore(trainData, labels);


		// Create an instance of the PCADimensionalityReduction class
		PCADimensionalityReduction pca;

		// Perform PCA and project the data onto a lower-dimensional space
		int num_dimensions = 2; // Number of dimensions to project onto
		std::vector<std::vector<double>> reduced_data = pca.performPCA(trainData, num_dimensions);

		MessageBox::Show("Run completed");
		return std::make_tuple(daviesBouldinIndex, silhouetteScore, std::move(labels), std::move(reduced_data));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
	}
}