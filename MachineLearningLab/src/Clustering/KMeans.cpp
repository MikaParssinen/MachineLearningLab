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
#include <algorithm>
using namespace System::Windows::Forms; // For MessageBox


///  KMeans class implementation  ///

// KMeans function: Constructor for KMeans class.//
KMeans::KMeans(int numClusters, int maxIterations)
	: numClusters_(numClusters), maxIterations_(maxIterations) {}


// fit function: Performs K-means clustering on the given dataset and return the centroids of the clusters.//
void KMeans::fit(const std::vector<std::vector<double>>& data) {
	// Create a copy of the data to preserve the original dataset
	std::vector<std::vector<double>> normalizedData = data;

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

	//int num_clusters = 3;
	double cluster_num_function;
	std::vector<double> Range_values(data.size());

	int range = 0;
	while (range < Range_values.size());
	{
		Range_values[range] = range;
		range++;
	}
	
	srand(unsigned(time(NULL)));
	std::random_shuffle(Range_values.begin(), Range_values.end());
	//std::shuffle(Range_values.begin(), Range_values.end(), Range_values.size()-1);

	int i = 0;
	while (i <= numClusters_)
	{
		centroids_.push_back(normalizedData[Range_values[i]]);
		
	}

	std::vector<double> distances;
	
	for (int iterations = 0; iterations < maxIterations_; iterations++)
	{
		
		std::vector<int, double> cluster_labels;
		int j = 0;
		for (auto& X_Point : normalizedData)
		{
			
			distances.push_back(SimilarityFunctions::euclideanDistance(X_Point, centroids_[j]));
			cluster_num_function = returnSmallestDistanceValue(distances);
			cluster_labels.push_back(cluster_num_function);
			if (j < numClusters_)
			{
				j++;
			}
		}
		/*
		std::vector<double> cluster_indices;
		for (int i = 0; i < numClusters_; i++)
		{
			cluster_indices.push_back()
		}
		*/
		
	}

	
	/*
	for (type variable_name : array / vector_name)
	{
		loop statements
			...
	}
	*/
}

double KMeans::returnSmallestDistanceValue(std::vector<double> distances)
{
	double index = 1.0;

	for (int i = 0; i < distances.size(); i++)
	{
		if (distances[i] < distances[index])
		{
			index = i;
		}
	}

	return index;
}


//// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.//
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels;
	labels.reserve(data.size());
	
	/* Implement the following:
		--- Initialize the closest centroid and minimum distance to the maximum possible value
		--- Iterate through each centroid
		--- Calculate the Euclidean distance between the point and the centroid
		--- Add the closest centroid to the labels vector
    */
	
	// TODO
	return labels; // Return the labels vector

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