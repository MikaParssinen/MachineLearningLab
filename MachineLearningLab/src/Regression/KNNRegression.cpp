#include "KNNRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <random>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


                                                ///  KNNRegression class implementation  ///


/// KNNRegression function: Constructor for KNNRegression class, initializing the number of neighbors to use for prediction./// 
KNNRegression::KNNRegression(int k) : k_(k) {}


/// fit function: Fits the KNNRegression model with the given training data. ///
void KNNRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
	X_train_ = X_train;
	y_train_ = y_train;
}
std::vector<double> KNNRegression::predict(const std::vector<std::vector<double>>& X_test) const {
    std::vector<double> y_pred; // Lagra förutsagda värden för alla testdatapunkter
    y_pred.reserve(X_test.size()); // Reservera minne för y_pred för att undvika frekvent omallokering

    // Kontrollera om träningsdata är tom
    if (X_train_.empty() || y_train_.empty()) {
        throw std::runtime_error("Error: Empty training data.");
    }

    // För varje testdatapunkt
    for (const auto& x_test : X_test) {
        // Beräkna avståndet till alla träningsdatapunkter
        std::vector<std::pair<double, double>> distances; // (avstånd, värde) par
        for (size_t i = 0; i < X_train_.size(); ++i) {
            double distance = SimilarityFunctions::euclideanDistance(x_test, X_train_[i]);
            distances.emplace_back(distance, y_train_[i]);
        }

        // Sortera avstånden i stigande ordning
        std::sort(distances.begin(), distances.end(),
            [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                return a.first < b.first;
            });

        // Hämta de närmaste K grannarna
        int k = k_; // Använd den K som definierades vid konstruktion
        double sumOfValues = 0.0;
        for (int i = 0; i < k; ++i) {
            sumOfValues += distances[i].second;
        }

        // Beräkna genomsnittet av värdena för k-närmaste grannar
        double predictedValue = sumOfValues / k;

        y_pred.push_back(predictedValue);
    }

    return y_pred; // Returnera vektorn med förutsagda värden för alla testdatapunkter
}


	/* Implement the following:
		--- Loop through each test data point
		--- Calculate Euclidean distance between test data point and each training data point
		--- Loop through the labels and their counts to find the most frequent label
		--- Store sum of y_train values for k-nearest neighbors
		--- Calculate average of y_train values for k-nearest neighbors
	*/





/// runKNNRegression: this function runs the KNN regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    KNNRegression::runKNNRegression(const std::string& filePath, int trainingRatio) {
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
