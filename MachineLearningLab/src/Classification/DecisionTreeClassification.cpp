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




// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;

	// Traverse the decision tree for each input vector and make predictions
	for (size_t i = 0; i < X.size(); ++i) {
		double prediction = traverseTree(X[i], root);  // Assuming 'root' is the root of the decision tree
		predictions.push_back(prediction);
	}

	return predictions;
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
//std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
//	std::vector<double> predictions;
//	
//
//	
//	for (const std::vector<double>& input : X) {
//		Node* currNode = root;
//
//		while (!currNode->isLeafNode())
//		{
//			if (input[currNode->feature] <= currNode->threshold)
//			{
//				//Move to left child
//				currNode = currNode->left;
//			}
//			else
//			{
//				//Move to right child
//				currNode = currNode->right;
//				
//			}
//
//		}
//	}
//
//
//	
//	return predictions;
//}


//Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
//	int num_samples = X.size();
//	int num_features = X[0].size();
//
//	// Define stopping criteria (for example, if maximum depth or minimum samples is reached)
//	if (depth >= max_depth || num_samples < min_samples_split || X.empty() || y.empty()) {
//		double leaf_value = mostCommonLabel(y); // Assign the most common label as the leaf value
//		return new Node(-1, -1, nullptr, nullptr, leaf_value);
//	}
//
//	double best_gain = -1.0;
//	int best_split_index = -1;
//	double best_split_thresh = -1.0;

	// Loop through each feature and find the best split
//	for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
//		// TODO: Calculate the best split for the current feature and update best_gain, best_split_index, and best_split_thresh
//	}
//
//	// If no split improves entropy (information gain is zero), create a leaf node
//	if (best_gain == 0.0) {
//		double leaf_value = mostCommonLabel(y); // Assign the most common label as the leaf value
//		return new Node(-1, -1, nullptr, nullptr, leaf_value);
//	}
//
//	// Split the data based on the best split
//	std::vector<std::vector<double>> X_left, X_right;
//	std::vector<double> y_left, y_right;
//
//	if (best_split_index < num_features && best_split_index >= 0) {
//		for (int i = 0; i < num_samples; ++i) {
//			if (X[i][best_split_index] <= best_split_thresh) {
//				X_left.push_back(X[i]);
//				y_left.push_back(y[i]);
//			}
//			else {
//				X_right.push_back(X[i]);
//				y_right.push_back(y[i]);
//			}
//		}
//	}
//
//	Node* left_child = nullptr;
//	Node* right_child = nullptr;
//
//	if (!X_left.empty() && !y_left.empty()) {
//		left_child = growTree(X_left, y_left, depth + 1);
//	}
//
//	if (!X_right.empty() && !y_right.empty()) {
//		right_child = growTree(X_right, y_right, depth + 1);
//	}
//
//	return new Node(best_split_index, best_split_thresh, left_child, right_child);
//}
//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	// Define stopping criteria
	if (depth >= max_depth || X.size() < min_samples_split) {
		return new Node(-1, -1, nullptr, nullptr, mostCommonLabel(y));
	}

	double best_gain = -1.0;
	int split_idx = -1;
	double split_thresh = -1.0;

	// Loop through candidate features and potential split thresholds.
	for (int i = 0; i < X[0].size(); ++i) {
		std::vector<double> X_column;
		for (int k = 0; k < X.size(); ++k) {
			X_column.push_back(X[k][i]);
		}
		for (int j = 0; j < X.size(); ++j) {
			double current_gain = informationGain(y, X_column, X[j][i]);
			if (current_gain > best_gain) {
				best_gain = current_gain;
				split_idx = i;
				split_thresh = X[j][i];
			}
		}
	}

	// Check if no split improves the gain (leaf node)
	if (best_gain <= 0.0) {
		return new Node(-1, -1, nullptr, nullptr, mostCommonLabel(y));
	}

	// Manually split the data based on the chosen threshold
	std::vector<std::vector<double>> X_left, X_right;
	std::vector<double> y_left, y_right;

	for (int i = 0; i < X.size(); ++i) {
		if (X[i][split_idx] < split_thresh) {
			X_left.push_back(X[i]);
			y_left.push_back(y[i]);
		}
		else {
			X_right.push_back(X[i]);
			y_right.push_back(y[i]);
		}
	}

	// Grow the children that result from the split
	Node* left = growTree(X_left, y_left, depth + 1); // grow the left tree
	Node* right = growTree(X_right, y_right, depth + 1);  // grow the right tree

	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}




//Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
//	
//
//	// Define stopping criteria based on depth or other conditions
//	if (depth < 0 || X.empty() || y.empty()) {
//		// Stop growing and return a leaf node with the most common label
//		double leaf_value = mostCommonLabel(y);
//		return new Node(-1, -1, nullptr, nullptr, leaf_value);
//	}
//
//
//	double best_gain = -1.0;
//	int split_idx = -1;
//	double split_thresh = 0.0;
//
//	int num_features = X[0].size();
//
//	// Loop through candidate features and potential split thresholds
//	for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) {
//		// Get unique values for the current feature
//		std::vector<double> unique_values;
//		for (size_t i = 0; i < X.size(); ++i) {
//			if (std::find(unique_values.begin(), unique_values.end(), X[i][feature_idx]) == unique_values.end()) {
//				unique_values.push_back(X[i][feature_idx]);
//			}
//		}
//
//		// Sort unique values
//		std::sort(unique_values.begin(), unique_values.end());
//
//		// Try each unique value as a potential split threshold
//		for (size_t i = 1; i < unique_values.size(); ++i) {
//			double potential_threshold = (unique_values[i - 1] + unique_values[i]) / 2.0;
//
//			// Extract the current feature column
//			std::vector<double> X_column;
//			for (size_t j = 0; j < X.size(); ++j) {
//				X_column.push_back(X[j][feature_idx]);
//			}
//
//			// Split the data based on the current feature and potential threshold
//			std::vector<int> left_indices, right_indices;
//			for (size_t j = 0; j < X.size(); ++j) {
//				if (X[j][feature_idx] <= potential_threshold)
//					left_indices.push_back(j);
//				else
//					right_indices.push_back(j);
//			}
//
//			// Calculate information gain for the potential split
//			double potential_gain = informationGain(y, X_column, potential_threshold);
//
//			// Update best gain and split information if needed
//			if (potential_gain > best_gain) {
//				best_gain = potential_gain;
//				split_idx = feature_idx;
//				split_thresh = potential_threshold;
//			}
//		}
//	}
//
//	// Check if no split improves the gain (leaf node)
//	if (best_gain <= 0.0) {
//		return new Node(mostCommonLabel(y));
//	}
//
//	// Grow the children that result from the best split
//	std::vector<std::vector<double>> left_X, right_X;
//	std::vector<double> left_y, right_y;
//
//	for (size_t i = 0; i < X.size(); ++i) {
//		if (X[i][split_idx] <= split_thresh) {
//			left_X.push_back(X[i]);
//			left_y.push_back(y[i]);
//		}
//		else {
//			right_X.push_back(X[i]);
//			right_y.push_back(y[i]);
//		}
//	}
//
//	Node* left_child = growTree(left_X, left_y, depth);  // Increment depth for left subtree
//	Node* right_child = growTree(right_X, right_y, depth);  // Increment depth for right subtree
//
//	return new Node(split_idx, split_thresh, left_child, right_child);
//}
//


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
//Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
//	
//
//
//	/* Implement the following:
//		--- define stopping criteria
//    	--- Loop through candidate features and potential split thresholds.
//		--- greedily select the best split according to information gain
//		---grow the children that result from the split
//	*/
//
//	
//	
//	double best_gain = -1.0; // set the best gain to -1
//	int split_idx = NULL; // split index
//	double split_thresh = NULL; // split threshold
//	
//	 // TODO
//	
//	Node* left; // grow the left tree
//	Node* right;  // grow the right tree
//	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
//}

double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	
	// Parent entropy
	double parent_entropy = EntropyFunctions::entropy(y); // Calculate whole entropy

	// Split the data based on the given split threshold
	std::vector<int> left_indices;
	std::vector<int> right_indices;

	for (size_t i = 0; i < X_column.size(); ++i) {
		if (X_column[i] <= split_thresh) {
			left_indices.push_back(i);
		}
		else {
			right_indices.push_back(i);
		}
	}

	// Calculate the weighted average of entropy for the children
	double left_entropy = EntropyFunctions::entropy(y, left_indices);
	double right_entropy = EntropyFunctions::entropy(y, right_indices);
	int total_samples = y.size();
	double weighted_child_entropy = (left_indices.size() * left_entropy + right_indices.size() * right_entropy) / total_samples;

	// Calculate information gain
	double information_gain = parent_entropy - weighted_child_entropy;

	return information_gain;
}


/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
//double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
//	//parent loss // You need to caculate entropy using the EntropyFunctions class//
//	double parent_entropy = EntropyFunctions::entropy(y);
//
//
//
//
//	/* Implement the following:
//	   --- generate split
//	   --- compute the weighted avg. of the loss for the children
//	   --- information gain is difference in loss before vs. after split	
//	*/
//
//	//for (int i = 0; i < y.size(); i++)
//	//{
//	//	if (X_column[i] <= split_thresh)
//	//	{
//	//		std::vector<int> leftVector;
//	//		leftVector.push_back(i);
//	//	}
//	//	else
//	//	{
//	//		std::vector<int> rightVector;
//	//		rightVector.push_back(i);
//	//	}
//	//}
//
//
//	double infogain = 0.0;
//	
//	// TODO
//
//	
//	return infogain;
//
//
//
//}



double DecisionTreeClassification::mostCommonLabel(std::vector<double>& y) {
	std::unordered_map<double, int> label_counts;

	// Räkna förekomst av varje klass
	for (const double& label : y) {
		label_counts[label]++;
	}

	double most_common = -1;  // Initialisera med ogiltig värde
	int max_count = 0;

	// Hitta den mest förekommande klassen
	for (const auto& pair : label_counts) {
		if (pair.second > max_count) {
			max_count = pair.second;
			most_common = pair.first;
		}
	}

	return most_common;
}





// mostCommonlLabel function: Finds the most common label in a vector of labels.//
//double DecisionTreeClassification::mostCommonLabel(std::vector<double>& y) {	
//	double most_common = 0.0;
//	
//	// TODO
//	return most_common;
//}
 

double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {
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






// traverseTree function: Traverses a decision tree given an input vector and a node.//
//double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {
//
//	/* Implement the following:
//		--- If the node is a leaf node, return its value
//		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
//		--- Otherwise, traverse the right subtree
//	*/
//	 //TODO
//	
//	return 0.0;
//}


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