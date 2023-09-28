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


bool DecisionTreeClassification::allSamplesHaveSameClass(std::vector<double>& y)
{
    if (y.empty())
    {
        return true;
    }

    double first_class = y[0];

    for (int i = 0; i < y.size(); i++)
    {
        if (y[i] != first_class)
        {
            return false;
        }
    }

    return true;
}



Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	
	
	//STEG 1
	
	// Define stopping criteria
	// Kontrollera om vi n�tt maximalt djup eller antalet prov �r f�r f�
	if (depth >= max_depth || X.size() < min_samples_split || allSamplesHaveSameClass(y) == true ) {
		return new Node(-1, -1, nullptr, nullptr, mostCommonLabel(y)); //Skapar lövnod som representerar den mest förekommande etikett.
	}


	//STEG 2:

	double best_gain = -1.0; 
	int split_idx = -1;
	double split_thresh = -1.0;



	// Loop through candidate features and potential split thresholds.
	
	// Loopa igenom varje attribut/längd/egenskap och hitta b�sta uppdelning
	for (int i = 0; i < X[0].size(); ++i) {
		std::vector<double> X_column;
		for (int k = 0; k < X.size(); ++k) {
			X_column.push_back(X[k][i]);
		}
		for (int j = 0; j < X.size(); ++j) {
			// Greedily select the best split according to information gain
			double current_gain = informationGain(y, X_column, X[j][i]);
			if (current_gain > best_gain) {
				best_gain = current_gain;
				split_idx = i; 
				split_thresh = X[j][i];
			}
		}
	}

	// Om ingen uppdelning f�rb�ttrar vinsten, skapa en l�vnode
	if (best_gain <= 0.0) {
		return new Node(-1, -1, nullptr, nullptr, mostCommonLabel(y));
	}

	// Split data on the best feature/threshold obtained from the above greedy search
	
	 // Dela upp datan baserat p� vald tr�skel
	std::vector<std::vector<double>> X_left, X_right;
	std::vector<double> y_left, y_right;

	for (int i = 0; i < X.size(); ++i) {
		if (X[i][split_idx] <= split_thresh) {
			X_left.push_back(X[i]);
			y_left.push_back(y[i]);
		}
		else {
			X_right.push_back(X[i]);
			y_right.push_back(y[i]);
		}
	}

	// Grow the children that result from the split
	
	// Skapa och v�xtr�det f�r de tv� delarna
	Node* left = growTree(X_left, y_left, depth + 1);
	Node* right = growTree(X_right, y_right, depth + 1);

	// Returnera en ny nod med uppdelningsinformationen
	return new Node(split_idx, split_thresh, left, right);
}

double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	
	// Parent entropy
	double parent_entropy = EntropyFunctions::entropy(y); // Calculate whole entropy

	// Split the data based on the given split threshold
	std::vector<int> left_indices;  
	std::vector<int> right_indices; 

	for (size_t i = 0; i < X_column.size(); ++i)  {
		if (X_column[i] <= split_thresh) {
			left_indices.push_back(i); // Add the index to the left child
		}
		else {
			right_indices.push_back(i); // Add the index to the right child
		}
	}

	// Calculate the weighted average of entropy for the children
	double left_entropy = EntropyFunctions::entropy(y, left_indices); 
	double right_entropy = EntropyFunctions::entropy(y, right_indices);
	int total_samples = y.size(); // Total number of samples
	double weighted_child_entropy = (left_indices.size() * left_entropy + right_indices.size() * right_entropy) / total_samples;  

	// Calculate information gain
	double information_gain = parent_entropy - weighted_child_entropy; 

	return information_gain; // Return the information gain
}

/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
//double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
//	//parent loss // You need to caculate entropy using the EntropyFunctions class//
//	double parent_entropy = EntropyFunctions::entropy(y);
//
//	/* Implement the following:
//	   --- generate split
//	   --- compute the weighted avg. of the loss for the children
//	   --- information gain is difference in loss before vs. after split	
//	*/

//	double infogain = 0.0;
//	
//	// TODO

//	return infogain;

//}

// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonLabel(std::vector<double>& y) {
	std::unordered_map<double, int> label_counts;

	// R�kna f�rekomst av varje klass
	for (const double& label : y) {
		label_counts[label]++;
	}

	double most_common = -1;  // Initialisera med ogiltig v�rde
	int max_count = 0;

	// Hitta den mest f�rekommande klassen
	for (const auto& pair : label_counts) {
		if (pair.second > max_count) {
			max_count = pair.second;
			most_common = pair.first;
		}
	}

	return most_common;
}



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



// traverseTree  function: Traverses a decision tree given an input vector and a node.//
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

