#include "KMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 

#include <cfloat> 

#include <numeric> 
using namespace System::Windows::Forms; // For MessageBox


///  KMeans class implementation  ///

// KMeans function: Constructor for KMeans class.//
KMeans::KMeans(int numClusters, int maxIterations)
	: numClusters_(numClusters), maxIterations_(maxIterations) {}





// fit function: Performs K-means clustering on the given dataset and return the centroids of the clusters.//
void KMeans::fit(const std::vector<std::vector<double>>& data) {
	// Create a copy of the data to preserve the original dataset
	std::vector<std::vector<double>> normalizedData = data;
	
	
	
	/*    	Initialize centroids randomly and  Randomly select unique centroid indices          */
	
	// Create a vector of indices from 0 to data.size() - 1
	std::vector<size_t> indexList(data.size()); // Create a vector of size data.size()
	std::iota(indexList.begin(), indexList.end(), 0); // Fill with 0, 1, ..., data.size() - 1

	// Shuffle the index list
	std::random_device rd; // Obtain a random number from hardware
	std::mt19937 gen(rd()); // Seed the generator
	std::shuffle(indexList.begin(), indexList.end(), gen); // Shuffle the indices

	centroids_.clear();  // Clear any existing centroids

	// Choose the first numClusters_ indices as initial centroids
	for (size_t i = 0; i < numClusters_; ++i) 
	{
		centroids_.push_back(normalizedData[indexList[i]]);  // Use the data point as a centroid
	}

	

	/*    	Perform K-means clustering until convergence or maxIterations is reached          */
	
	
	bool centroidsChanged = true;
	int iter = 0;

	while (centroidsChanged && iter < maxIterations_) {
		std::vector<std::vector<double>> oldCentroids = centroids_; // Save the old centroids
		double minDistance = DBL_MAX; // Initialize minDistance to the maximum value of a double
		std::vector<std::vector<double>> clusterSums(numClusters_, std::vector<double>(data[0].size(), 0)); // Initialize clusterSums to 0
		std::vector<int> clusterCounts(numClusters_, 0); // Initialize clusterCounts to 0

		// Tilldela datapunkter till närmaste centroid
		for (const auto& point : normalizedData) 
		{
			
			int closestCentroidIndex = -1; // Index of the closest centroid
			std::vector<std::pair<int, double>> closestCentroids(normalizedData.size(), { -1, DBL_MAX }); // {Index för närmaste centroid, Avstånd till närmaste centroid}

			for (size_t i = 0; i < normalizedData.size(); ++i)
			{
				for (size_t j = 0; j < numClusters_; ++j)
				{
					double distance = SimilarityFunctions::euclideanDistance(normalizedData[i], centroids_[j]); // Calculate the Euclidean distance between the point and the current centroid 
					if (distance < closestCentroids[i].second) {
						closestCentroids[i] = { static_cast<int>(j), distance };
					}
				}
			}

			// Lägg till datapunkter i respektive kluster baserat på den närmaste centroiden
			std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(data[0].size(), 0));
			std::vector<int> clusterCounts(numClusters_, 0);

			for (size_t i = 0; i < normalizedData.size(); ++i) 
			{
				int closestCentroidIndex = closestCentroids[i].first;
				for (size_t j = 0; j < normalizedData[i].size(); ++j)
				{
					newCentroids[closestCentroidIndex][j] += normalizedData[i][j];
				}
				clusterCounts[closestCentroidIndex]++;
			}

			// Uppdatera centroiderna
			for (size_t i = 0; i < numClusters_; ++i) 
			{
				for (size_t j = 0; j < data[0].size(); ++j) 
				{
					if (clusterCounts[i] > 0)
						centroids_[i][j] = newCentroids[i][j] / clusterCounts[i];
				}
			}

			// Kontrollera om centroiderna har förändrats
			centroidsChanged = !areCentroidsEqual(oldCentroids, centroids_);
			iter++;
		}
	}
}


/* Implement the following:
		---	Initialize centroids randomly
		--- Randomly select unique centroid indices
		---	Perform K-means clustering
		--- Assign data points to the nearest centroid
		--- Calculate the Euclidean distance between the point and the current centroid
		--- Update newCentroids and clusterCounts
		--- Update centroids
		---  Check for convergence
	*/

	// TODO


//// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.//
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels;
	labels.reserve(data.size());

	for (const auto& point : data) {
		int closestCentroid = -1; // Initialize the closest centroid to an invalid value
		double minDistance = DBL_MAX; // Initialize the minimum distance to the maximum possible value

		for (size_t i = 0; i < numClusters_; ++i) {
			double distance = SimilarityFunctions::euclideanDistance(point, centroids_[i]); // Calculate the Euclidean distance between the point and the centroid
			if (distance < minDistance) {
				minDistance = distance;
				closestCentroid = static_cast<int>(i); // Update the closest centroid index
			}
		}

		labels.push_back(closestCentroid+1); // Add the closest centroid to the labels vector
	}

	return labels; // Return the labels vector
}





bool KMeans::areCentroidsEqual(const std::vector<std::vector<double>>& centroids1, const std::vector<std::vector<double>>& centroids2) const {
	if (centroids1.size() != centroids2.size())
		return false;

	for (size_t i = 0; i < centroids1.size(); ++i) {
		if (centroids1[i] != centroids2[i])
			return false;
	}

	return true;
}


/// runKMeans: this function runs the KMeans clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
KMeans::runKMeans(const std::string& filePath) {
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
