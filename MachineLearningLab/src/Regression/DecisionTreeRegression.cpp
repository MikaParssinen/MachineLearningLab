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



///  DecisionTreeRegression class implementation  ///


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



// growTree function: Grows a decision tree regression model using the given data and parameters //
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {


	int split_idx = -1;
	double split_thresh = 0.0;
	
	if (depth >= max_depth) //H�r ska vi ut�ka conditionals. 
		//regressionen g�r p� labeln baserat p� siffror. En vettig conditional �r att se ifall siffrorna i vectorn �r n�golunda samma vilket kan g�ra att vi d� kan g�ra en leaf node
		//exempelvis om 2,4,6,4 finns i vectorn blir produkten 16 och mean blir 4. h�r ska vi forts�tta skapa decision nodes
		//MEN: om exempelvis 2,2,3,2,1 finns i vectorn blir produkten 10 och mean = 2 vilket kan g�ra att vi kan skapa en leaf d� mean �r samma som majoriteten av siffrorna i vectorn. 
	{

	}

	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- Find the best split threshold for the current feature.
		--- grow the children that result from the split
	*/
	
	// TODO

	Node* left;
	Node* right;
	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {

	double num_of_samples = y.size(); //variabeln s�tts till storleken av vector y
	double mse = 0.0;

	for (int i = 0; i < num_of_samples; i++) //j�mf�r f�r alla samples i y med alla features i x och v�r input parameter split_thresh
	{
		double y_value = y[i];//variabeln s�tts till v�rdet av y[i]
		double feature_comparison = (X_column[i] <= split_thresh) ? 0.0 : 1.0;//"feature_comparison" kommer att f� antingen v�rdet 0.0 eller 1.0 beroende p� ifall v�rdet av x_column[i] �r mindre eller st�rre �n split_thresh
		double error_value = (y_value - feature_comparison);//error_value blir lika med differensen av y_value och feature comparison. Allts� v�rat error v�rde
		mse += error_value * error_value;//enligt formeln f�r MSE (mean squared error) s�tter vi error v�rdet upph�jt till tv�
	}
    
	mse = mse / num_of_samples;//returnerar mse efter vi tagit fram mean
	
	return mse;
}

// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

	double counter;
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

