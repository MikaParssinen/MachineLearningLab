#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map> 

using namespace System::Windows::Forms; // For MessageBox


                                            ///  LogisticRegression class implementation  ///
// Constractor

LogisticRegression::LogisticRegression(double learning_rate, int num_epochs)
    : learning_rate(learning_rate), num_epochs(num_epochs) {}

// Fit method for training the logistic regression model
//void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
//    int num_features = X_train[0].size();
//    int num_classes = std::set<double>(y_train.begin(), y_train.end()).size();
//
//
//	/* Implement the following:
//       	--- Initialize weights for each class
//    	--- Loop over each class label
//    	--- Convert the problem into a binary classification problem
//        --- Loop over training epochs
//       	--- Add bias term to the training example
//    	--- Calculate weighted sum of features
//        --- Calculate the sigmoid of the weighted sum
//        --- Update weights using gradient descent
//    */
//    
//    // TODO
//}

void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    int num_samples = X_train.size();
    int num_features = X_train[0].size();
    int num_classes = std::set<double>(y_train.begin(), y_train.end()).size();

    // Initialize weights for each class
    std::vector<std::vector<double>> weights(num_classes, std::vector<double>(num_features + 1, 0.0)); // +1 for bias

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            // Add bias term to the training example
            std::vector<double> input_with_bias = X_train[i];
            input_with_bias.push_back(1.0); // Adding bias

            // Convert the problem into a binary classification problem for each class
            for (int c = 0; c < num_classes; c++) {
                int y_binary = (y_train[i] == c) ? 1 : 0; // 1 if class matches, 0 otherwise

                // Calculate weighted sum of features
                double weighted_sum = 0.0;
                for (int j = 0; j < num_features + 1; j++) {
                    weighted_sum += weights[c][j] * input_with_bias[j];
                }

                // Calculate the sigmoid of the weighted sum
                double sigmoid_result = 1.0 / (1.0 + exp(-weighted_sum));

                // Calculate the error (difference between predicted and actual)
                double error = sigmoid_result - y_binary;

                // Update weights using gradient descent
                for (int j = 0; j < num_features + 1; j++) {
                    // Calculate the gradient for this weight
                    double gradient = error * input_with_bias[j];

                    // Update the weight using gradient descent
                    weights[c][j] -= learning_rate * gradient;
                }
            }
        }
    }

    // Store the computed weights for prediction
    this->weights = weights;
}




std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;

    for (const auto& input : X_test) {
        std::vector<double> input_with_bias = input;
        input_with_bias.push_back(1.0); // Adding bias

        // Initialize max_score and predicted_class appropriately for each input
        double max_score = -std::numeric_limits<double>::infinity();
        int predicted_class = -1;

        // Calculate scores for each class by computing the weighted sum of features
        for (size_t c = 0; c < weights.size(); c++) {
            double score = 0.0;
            for (size_t j = 0; j < weights[c].size(); j++) {
                score += weights[c][j] * input_with_bias[j];
            }

            // Predict class label with the highest score
            if (score > max_score) {
                max_score = score;
                predicted_class = static_cast<int>(c);  // Convert size_t to int
            }
        }

        predictions.push_back(predicted_class);
    }

    return predictions;
}


//
//// Predict method to predict class labels for test data
//std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
//    std::vector<double> predictions;
//    
//    /* Implement the following:
//    	--- Loop over each test example
//        --- Add bias term to the test example
//        --- Calculate scores for each class by computing the weighted sum of features
//        --- Predict class label with the highest score
//    */
//      
//    // TODO
//    
//    return predictions;
//}

/// runLogisticRegression: this function runs the logistic regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> 
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {

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
        fit(trainData, trainLabels);

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