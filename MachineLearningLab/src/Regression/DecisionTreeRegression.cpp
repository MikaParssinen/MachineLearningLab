#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>
using namespace System::Windows::Forms; // For MessageBox





// Constructor for DecisionTreeRegression class.//
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr)
{
}


// fit function:Fits a decision tree regression model to the given data.//
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}

// predict function:Traverses the decision tree and returns the predicted value for a given input vector.//
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;
	for (size_t i = 0; i < X.size(); ++i) {
		double prediction = traverseTree(X[i], root);  // Assuming 'root' is the root of the decision tree
		predictions.push_back(prediction);
	}
	return predictions;
}

Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	int num_samples = y.size();
	int num_features = n_feats;

	// Check for stopping criteria: If max depth is reached or the number of samples is less than min_samples_split
	if (depth >= max_depth || num_samples < min_samples_split) {
		// Create a leaf node and set its value as the mean of y
		return new Node(-1, -1, nullptr, nullptr, mean(y));
	}

	
	double best_mse = DBL_MAX;
	int best_split_idx = -1;
	double best_split_thresh = 0.0;
	/*std::vector<double> best_left_labels; 
	std::vector<double> best_right_labels;*/


	// Loopa igenom varje attribut/längd/egenskap och hitta b�sta uppdelning
	for (int i = 0; i < X[0].size(); ++i) {
		std::vector<double> X_column;
		for (int k = 0; k < X.size(); ++k) {
			X_column.push_back(X[k][i]);
		}
		for (int j = 0; j < X.size(); ++j) {
			double current_mse = meanSquaredError(y, X_column, X[j][i]);
			if (current_mse < best_mse) {
				best_mse = current_mse;
				best_split_idx = i;
				best_split_thresh = X[j][i];
			}
		}
	}

	// Check if no split improves MSE
	/*if (best_mse == initial_mse) {
		return new Node(-1, -1, nullptr, nullptr, mean(y));
	}*/

	// Dela upp datan baserat p� vald tr�skel
	std::vector<std::vector<double>> X_left, X_right;
	std::vector<double> y_left, y_right;

	for (int i = 0; i < X.size(); ++i) {
		if (X[i][best_split_idx] <= best_split_thresh) {
			X_left.push_back(X[i]);
			y_left.push_back(y[i]);
		}
		else {
			X_right.push_back(X[i]);
			y_right.push_back(y[i]);
		}
	}

	// Recursively grow the left and right subtrees
	Node* left_child = growTree(X_left, y_left, depth + 1);
	Node* right_child = growTree(X_right, y_right, depth + 1);

	// Create a decision node using the best split
	return new Node(best_split_idx, best_split_thresh, left_child, right_child);
}

//Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
//
//
//	int split_idx = -1;
//	double split_thresh = 0.0;
//
//
//	/* Implement the following:
//		--- define stopping criteria
//		--- Loop through candidate features and potential split thresholds.
//		--- Find the best split threshold for the current feature.
//		--- grow the children that result from the split
//	*/
//
//	// TODO
//
//	Node* left;
//	Node* right;
//	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
//}

//
//// growTree function: Grows a decision tree regression model using the given data and parameters //
//Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
//
//
//	int split_idx = -1;
//	double split_thresh = 0.0;
//	
//	if (depth >= max_depth) //Här ska vi utöka conditionals. 
//		//regressionen gör på labeln baserat på siffror. En vettig conditional är att se ifall siffrorna i vectorn är någolunda samma vilket kan göra att vi då kan göra en leaf node
//		//exempelvis om 2,4,6,4 finns i vectorn blir produkten 16 och mean blir 4. här ska vi fortsätta skapa decision nodes
//		//MEN: om exempelvis 2,2,3,2,1 finns i vectorn blir produkten 10 och mean = 2 vilket kan göra att vi kan skapa en leaf då mean är samma som majoriteten av siffrorna i vectorn. 
//	{
//
//	}
//
//	/* Implement the following:
//		--- define stopping criteria
//    	--- Loop through candidate features and potential split thresholds.
//		--- Find the best split threshold for the current feature.
//		--- grow the children that result from the split
//	*/
//	
//	// TODO
//
//	Node* left;
//	Node* right;
//	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
//}


/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	
	double num_of_samples = y.size();
	double mse = 0.0;
	double error_value = 0.0;

	double feature_comparison_left = 0.0;
	double feature_comparison_right = 0.0;

	std::vector<double>left;
	std::vector<double>right;
	for(int i = 0; i<y.size(); i++)
	{
		if(X_column[i] <= split_thresh)
		{
			left.push_back(y[i]);
		}
		else
		{
			right.push_back(y[i]);
		}
	}

	feature_comparison_left = mean(left);
	feature_comparison_right = mean(right);

	for(int i = 0; i < left.size(); i++)
	{
		mse += (left[i] - feature_comparison_left) * (left[i] - feature_comparison_left);
	}

	for(int i = 0; i<right.size(); i++)
	{
		mse += (right[i] - feature_comparison_right) * (right[i] - feature_comparison_right);
	}
	mse = mse / num_of_samples;
	return mse;
}


// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

	double counter = 0.0;
	double num_of_values = values.size();
	for (int i = 0; i < values.size(); i++)
	{
		counter += values[i];
	}
	double meanValue = 0.0;
	meanValue = (counter / num_of_values);
	
	return meanValue;
}

// traverseTree function: Traverses the decision tree and returns the predicted value for the given input vector.//
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {

	// If the node is a leaf node, return its value
	if (node->isLeafNode()) {
		return node->value;  // Assuming value is the predicted class label for the leaf node
	}

	// Check the feature value of the input vector
	double feature_value = x[node->feature];

	// If the feature value is less than or equal to the node's threshold, traverse the left subtree
	if (feature_value <= node->threshold) {
		return traverseTree(x, node->left);
	}
	// Otherwise, traverse the right subtree
	else {
		return traverseTree(x, node->right);
	}
	
}


/// runDecisionTreeRegression: this function runs the Decision Tree Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.

std::tuple<double, double, double, double, double, double,
	std::vector<double>, std::vector<double>,
	std::vector<double>, std::vector<double>>
	DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio) {
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
		// Load the dataset from the file path
		std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

		// Convert the dataset from strings to doubles
		std::vector<std::vector<double>> dataset;
		bool isFirstRow = true; // Flag to identify the first row

		for (const auto& row : data) {
			if (isFirstRow) {
				isFirstRow = false;
				continue; // Skip the first row (header)
			}

			std::vector<double> convertedRow;
			for (const auto& cell : row) {
				try {
					double value = std::stod(cell);
					convertedRow.push_back(value);
				}
				catch (const std::exception& e) {
					// Handle the exception or set a default value
					std::cerr << "Error converting value: " << cell << std::endl;
					// You can choose to set a default value or handle the error as needed
				}
			}
			dataset.push_back(convertedRow);
		}

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate evaluation metrics (e.g., MAE, MSE)
		double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
		double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
		double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate evaluation metrics for training data
		double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
		double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
		double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(test_mae, test_rmse, test_rsquared,
			train_mae, train_rmse, train_rsquared,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			std::vector<double>(), std::vector<double>(),
			std::vector<double>(), std::vector<double>());
	}
}

