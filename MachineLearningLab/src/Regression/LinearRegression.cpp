#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <cstdlib>   // For rand() function
#include <ctime>
using namespace System::Windows::Forms; // For MessageBox



										///  LinearRegression class implementation  ///


<<<<<<< Updated upstream
=======
double LinearRegression::generateRandomCoefficient() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    return dis(gen);
}


void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, int num_iterations, double learning_rate) {
    int numSamples = trainData.size();
    int numFeatures = trainData[0].size();

    // Check if training data and labels are of the same size
    if (numSamples != trainLabels.size()) {
        std::cerr << "Training data and labels are not of the same size" << std::endl;
        return;
    }

    // Initialize coefficients with random values
    m_coefficients.resize(numFeatures);
    for (int i = 0; i < numFeatures; i++) {
        m_coefficients[i] = generateRandomCoefficient();
    }

    // Gradient Descent
    for (int iter = 0; iter < num_iterations; iter++) {
        // Calculate predictions
        std::vector<double> predictions(numSamples, 0.0);
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                predictions[i] += m_coefficients[j] * trainData[i][j];
            }
        }

        // Compute the error (difference between predictions and actual labels)
        std::vector<double> error(numSamples);
        for (int i = 0; i < numSamples; i++) {
            error[i] = predictions[i] - trainLabels[i];
        }

        // Compute the gradient and update coefficients using gradient descent
        for (int j = 0; j < numFeatures; j++) {
            double gradient = 0.0;
            for (int i = 0; i < numSamples; i++) {
                gradient += error[i] * trainData[i][j];
            }
            gradient /= numSamples;
            m_coefficients[j] -= learning_rate * gradient;
        }
    }

}
// This implementation is using Matrix Form method
/* Implement the following:
    --- Check if the sizes of trainData and trainLabels match
    --- Convert trainData to matrix representation
    --- Construct the design matrix X
    --- Convert trainLabels to matrix representation
    --- Calculate the coefficients using the least squares method
    --- Store the coefficients for future predictions
*/

// TODO



>>>>>>> Stashed changes
// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {

	// This implementation is using Matrix Form method
	/* Implement the following:	  
	    --- Check if the sizes of trainData and trainLabels match
	    --- Convert trainData to matrix representation
	    --- Construct the design matrix X
		--- Convert trainLabels to matrix representation
		--- Calculate the coefficients using the least squares method
		--- Store the coefficients for future predictions
	*/
	
	// TODO
}
<<<<<<< Updated upstream


// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {

	// This implementation is using Matrix Form method    
    /* Implement the following
		--- Check if the model has been fitted
		--- Convert testData to matrix representation
		--- Construct the design matrix X
		--- Make predictions using the stored coefficients
		--- Convert predictions to a vector
	*/
	
	// TODO

	std::vector<double> result;
	
    return result;
=======
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {
    int numSamples = testData.size();
    int numFeatures = m_coefficients.size();

    std::vector<double> predictions(numSamples, 0.0);

    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            predictions[i] += m_coefficients[j] * testData[i][j];
        }
    }

    return predictions;
>>>>>>> Stashed changes
}

//// Function to make predictions on new data //
//std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {
//    if (m_coefficients.size() == 0) {
//		MessageBox::Show("Please fit the model to the training data first");
//		return {}; // Return an empty vector since the model hasn't been fitted yet
//	}
//
//    //Convert testData to matrix representation
//     Eigen::MatrixXd X(testData.size(), testData[0].size());
//     for (int i = 0; i < testData.size(); i++) {
//     	for (int j = 0; j < testData[0].size(); j++)
//     		//Construct the design matrix X
//     		X(i, j) = testData[i][j];
//     }
//     
//     //Make predictions using the stored coefficients
//     Eigen::MatrixXd Y = X * m_coefficients;
//     //Convert predictions to a vector
//     std::vector<double> result;
//     for (int i = 0; i < Y.rows(); i++) {
//     	result.push_back(Y(i, 0));
//     }
//	
//     // This implementation is using Matrix Form method    
//    /* Implement the following
//		--- Check if the model has been fitted
//		--- Convert testData to matrix representation
//		--- Construct the design matrix X
//		--- Make predictions using the stored coefficients
//		--- Convert predictions to a vector
//	*/
//	
//	// TODO
//
//
//    return result;
//}



/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio) {
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
        fit(trainData, trainLabels,10,0.0001);

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