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

    
   
    for (int i = 0; i < num_classes; i++)
    {
        // Initialize weights for each class
        std::vector<double> weight_vector(num_features + 1, 0.0); // +1 for bias
        for (int epoch = 0; epoch < num_epochs; epoch++)
        {
            
            std::vector<double> Gradient_vector(num_features + 1, 0.0);
            for (int samples = 0; samples < num_samples; samples++)
            {
                int y_binary = (y_train[samples] == i+1) ? 1 : 0; // 1 if class matches, 0 otherwise
                std::vector<double> input_with_bias = X_train[samples];
                input_with_bias.push_back(1.0); // Adding bias
                double weighted_sum = weight_vector[0];

                for (int features = 0; features < num_features; features++)
                {
                    weighted_sum += weight_vector[features + 1] * input_with_bias[features];
                }

                // Calculate the sigmoid of the weighted sum
                double sigmoid_result = sigmoid(weighted_sum);
                
                double error = sigmoid_result - y_binary;

                Gradient_vector[0] += error * 1.0; //Form av derivata
                
                for (int gradient_calc = 0; gradient_calc < num_features; gradient_calc++)
                {
                    Gradient_vector[gradient_calc + 1] += error * input_with_bias[gradient_calc];
                }

            }


            for (int w = 0; w <= num_features; w++)
            {
                weight_vector[w] -= learning_rate * (Gradient_vector[w] / num_samples);
             
            }


            


        }

        weights.push_back(weight_vector);
    }

    /*
   // Loop over training epochs
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            // Add bias term to the training example
            std::vector<double> input_with_bias = X_train[i];
            input_with_bias.push_back(1.0); // Adding bias

            // Convert the problem into a binary classification problem for each class
            for (int c = 0; c < num_classes; c++) {
                int y_binary = (y_train[i] == c) ? 1 : 0; // 1 if class matches, 0 otherwise

                // Calculate weighted sum of features
                double weighted_sum = weights[c][0];
                for (int j = 0; j < num_features; j++) {

                    weighted_sum += weights[c][j+1] * input_with_bias[j];
                }

                // Calculate the sigmoid of the weighted sum
                double sigmoid_result = sigmoid(weighted_sum);
               
                
                // double sigmoid_result = 1.0 / (1.0 + exp(-weighted_sum));

                // Calculate the error (difference between predicted and actual)
                double error = sigmoid_result - y_binary;
                 
                double gradient = error;
                weights[c][0] -= learning_rate * gradient;


                /*
                for (int j = 0; j < num_features + 1; ++j) {
                    weights[c][j] -= learning_rate * (error) * input_with_bias[j];
                }
                */

                // Update weights using gradient descent
                
                /*
                for (int j = 0; j < num_features; j++) {
                    // Calculate the gradient for this weight
                    gradient = error * input_with_bias[j];

                    // Update the weight using gradient descent
                    weights[c][j+1] -= learning_rate * gradient;
                }
                
            }
        }
    }

    // Store the computed weights for prediction
    this->weights = weights;
    */
}




std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;

    for (const auto& input : X_test) {
        //std::vector<double> input_with_bias = input;
        //input_with_bias.push_back(1.0); // Adding bias

        // Initialize max_score and predicted_class appropriately for each input
        double max_score = -std::numeric_limits<double>::infinity();
        int predicted_class = -1;

        // Calculate scores for each class by computing the weighted sum of features
        for (size_t c = 0; c < weights.size(); c++) {
            double score = weights[c][0];
            for (size_t j = 0; j < weights[c].size()-1; j++) {
                score += weights[c][j+1] * input[j];
            }

            // Predict class label with the highest score
            score = sigmoid(score);
            if (score > max_score) {
                max_score = score;
                predicted_class = static_cast<int>(c+1);  // Convert size_t to int
                
                
                
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