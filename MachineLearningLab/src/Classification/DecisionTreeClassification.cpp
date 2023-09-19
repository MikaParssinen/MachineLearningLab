#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include "../DataUtils/DataPreprocessor.h"
using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;

    for (const auto& sample : X) {
        double prediction = traverseTree(sample, root);
        predictions.push_back(prediction);
    }

    return predictions;
}




// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
//std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	//std::vector<double> predictions;
	
	// Implement the function
	// TODO
	
	//return predictions;
//}

Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
    int num_samples = X.size();
    int num_features = X[0].size();

    // Define stopping criteria (for example, if maximum depth or minimum samples is reached)
    if (depth >= max_depth || num_samples < min_samples_split) {
        double leaf_value = mostCommonLabel(y); // Assign the most common label as the leaf value
        return new Node(-1, -1, nullptr, nullptr, leaf_value);
    }

    double best_gain = -1.0;
    int best_split_index = -1;
    double best_split_thresh = -1.0;

    // Loop through each feature and find the best split
    for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        // TODO: Calculate the best split for the current feature and update best_gain, best_split_index, and best_split_thresh
    }

    // If no split improves entropy (information gain is zero), create a leaf node
    if (best_gain == 0.0) {
        double leaf_value = mostCommonLabel(y); // Assign the most common label as the leaf value
        return new Node(-1, -1, nullptr, nullptr, leaf_value);
    }

    // Split the data based on the best split
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<double> y_left, y_right;

    for (int i = 0; i < num_samples; ++i) {
        if (X[i][best_split_index] <= best_split_thresh) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    // Recursively grow the left and right subtrees
    Node* left_child = growTree(X_left, y_left, depth + 1);
    Node* right_child = growTree(X_right, y_right, depth + 1);

    return new Node(best_split_index, best_split_thresh, left_child, right_child);
}


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
//Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	
	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- greedily select the best split according to information gain
		---grow the children that result from the split
	*/
	
	//double best_gain = -1.0; // set the best gain to -1
	//int split_idx = NULL; // split index
	//double split_thresh = NULL; // split threshold
	
	// TODO
	
	//Node* left; // grow the left tree
	//Node* right;  // grow the right tree
	//return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
//}

double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
    // Calculate parent entropy
    double parent_entropy = EntropyFunctions::entropy(y);

    int total_samples = y.size();
    int left_samples = 0, right_samples = 0;
    std::vector<double> y_left, y_right;

    // Split labels based on the threshold
    for (size_t i = 0; i < total_samples; ++i) {
        if (X_column[i] <= split_thresh) {
            y_left.push_back(y[i]);
            ++left_samples;
        } else {
            y_right.push_back(y[i]);
            ++right_samples;
        }
    }

    // Calculate the weighted average of the entropy for the children
    double left_entropy = (static_cast<double>(left_samples) / total_samples) * EntropyFunctions::entropy(y_left);
    double right_entropy = (static_cast<double>(right_samples) / total_samples) * EntropyFunctions::entropy(y_right);

    // Information gain is the difference in entropy before vs. after split
    double ig = parent_entropy - ((left_samples / static_cast<double>(total_samples)) * left_entropy +
                                    (right_samples / static_cast<double>(total_samples)) * right_entropy);

    return ig;
}

/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
//double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// parent loss // You need to caculate entropy using the EntropyFunctions class//
	//double parent_entropy = EntropyFunctions::entropy(y);

	/* Implement the following:
	   --- generate split
	   --- compute the weighted avg. of the loss for the children
	   --- information gain is difference in loss before vs. after split
	*/
	//double ig = 0.0;
	
	// TODO
	
	//return ig;
//}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {	
	double most_common = 0.0;
	
	// TODO
	return most_common;
}


// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {

	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/
	// TODO
	
	return 0.0;
}


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
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

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);//

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate accuracy using the true labels and predicted labels for the test data
		double test_accuracy = Metrics::accuracy(testLabels, testPredictions);


		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate accuracy using the true labels and predicted labels for the training data
		double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(train_accuracy, test_accuracy,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<double>(),
			std::vector<double>(), std::vector<double>(),
			std::vector<double>());
	}
}