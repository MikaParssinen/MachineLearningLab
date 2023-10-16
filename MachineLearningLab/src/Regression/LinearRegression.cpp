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
#include <float.h>
using namespace System::Windows::Forms; // For MessageBox



										///  LinearRegression class implementation  ///


void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, int num_iterations, double learning_rate) {

  
    double num_feats = trainData[0].size();
    int num_samples = trainData.size();

    m_coefficients = Eigen::VectorXd::Ones(num_feats + 1);

    //Check if training data and labels are of the same size
    if (trainData.size() == trainLabels.size()) {

    
        int num_samples = trainData.size();
        int num_features = trainData[0].size();
        
        
        
        m_coefficients = Eigen::VectorXd::Ones(num_features + 1);

        for (int i = 0, constant = 1; i < m_coefficients.size(); i++)
        {
            m_coefficients[i] = constant;
        }
        
        double predicted = m_coefficients[0];
        std::vector<std::vector<double>> normalized_vector;
        std::vector<double> min;
        std::vector<double> max;
        double temp_min = DBL_MAX;
        double temp_max = DBL_MIN;

        for (int i = 0; i < num_features; i++)
        {
            
            for (int j = 0; j < num_samples; j++)
            {
                if (trainData[j][i] < temp_min)
                {
                    temp_min = trainData[j][i];
                }
                if (trainData[j][i] > temp_max  )
                {
                    temp_max = trainData[j][i];
                }


            }
            max.push_back(temp_max);
            min.push_back(temp_min);
        }

        for (int i = 0; i < trainData.size(); i++)
        {
            std::vector<double> norm_temp;
            for (int j = 0; j < num_features; j++)
            {
               double normalized_value = (trainData[i][j] - min[j]) / (max[j] - min[j]);
               norm_temp.push_back(normalized_value);
            }
            normalized_vector.push_back(norm_temp);
        }
       
        for (int epoch = 0; epoch < num_iterations; epoch++)
        {
            std::vector<double> gradient(num_features + 1, 0.0);
            for (int samples = 0; samples < num_samples; samples++) {


                
                double predicted = m_coefficients[0];
                for (int features = 1; features <= num_features; features++)
                {
                    predicted += m_coefficients[features] * normalized_vector[samples][features-1];
                    
                }
                //predicted += m_coefficients[num_features]; // Intercept term

                // Calculate the error for the current sample
                double error = trainLabels[samples] - predicted;
                
                gradient[0] += error * 1.0;
                for (int gradient_calc = 1; gradient_calc <= num_features; gradient_calc++) {
                    gradient[gradient_calc] += error * normalized_vector[samples][gradient_calc-1];
                }
                
                //for (int gradient_calc = 0; gradient_calc < num_features; gradient_calc++)
                //{
                    
                //}

            }
            //m_coefficients -= learning_rate * gradient / num_samples;
            for (int fill = 0; fill < gradient.size(); fill++)
            {
                gradient[fill] = (-2) * learning_rate * gradient[fill];
                m_coefficients[fill] -= gradient[fill] / num_samples;
            }
            
        }

 

    }
    else {
        MessageBox::Show("Training data and labels are not of the same size");
    }



    std::vector<double> K(trainData[0].size() + 1);

        for (int i = 0; i < trainData[0].size() + 1; i++)
        {
            K[i] = m_coefficients[i];
        }
    // This implementation is using Matrix Form method

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



std::vector<std::vector<double>> LinearRegression::NormalizeForPredict(const std::vector<std::vector<double>>& trainData)
{
    int num_features = trainData[0].size();
    int num_samples = trainData.size();


    double predicted = m_coefficients[0];
    std::vector<std::vector<double>> normalized_vector;
    std::vector<double> min;
    std::vector<double> max;
    double temp_min = DBL_MAX;
    double temp_max = DBL_MIN;

    for (int i = 0; i < num_features; i++)
    {

        for (int j = 0; j < num_samples; j++)
        {
            if (trainData[j][i] < temp_min)
            {
                temp_min = trainData[j][i];
            }
            if (trainData[j][i] > temp_max)
            {
                temp_max = trainData[j][i];
            }

        }
        max.push_back(temp_max);
        min.push_back(temp_min);
    }

    for (int i = 0; i < trainData.size(); i++)
    {
        std::vector<double> norm_temp;
        for (int j = 0; j < num_features; j++)
        {
            double normalized_value = (trainData[i][j] - min[j]) / (max[j] - min[j]);
            norm_temp.push_back(normalized_value);
        }
        normalized_vector.push_back(norm_temp);
    }

    return normalized_vector;
}


std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData, int gradient) {
    if (m_coefficients.size() == 0) {
        MessageBox::Show("Please fit the model to the training data first");
        return {}; // Return an empty vector since the model hasn't been fitted yet
    }

    //Convert testData to matrix representation


    std::vector<std::vector<double>> normalized_vector = NormalizeForPredict(testData);

    Eigen::MatrixXd X(normalized_vector.size(), normalized_vector[0].size()+1);
    for (int i = 0; i < normalized_vector.size(); i++) {
        X(i, 0) = 1.0;
        for (int j = 1; j < normalized_vector[0].size()+1; j++)
            //Construct the design matrix X
            X(i, j) = normalized_vector[i][j-1];
    }

    //Make predictions using the stored coefficients
    Eigen::VectorXd Y = X * m_coefficients;
    //Convert predictions to a vector
    std::vector<double> result;
    
    for (int i = 0; i < Y.rows(); i++) {
        result.push_back(Y(i));
    }

    // This implementation is using Matrix Form method    
   /* Implement the following
       --- Check if the model has been fitted
       --- Convert testData to matrix representation
       --- Construct the design matrix X
       --- Make predictions using the stored coefficients
       --- Convert predictions to a vector
   */

   // TODO


    return result;
}

//
//std::vector<double> linearregression::predict(const std::vector<std::vector<double>>& testdata) {
//    if (m_coefficients.size() == 0) {
//		messagebox::show("please fit the model to the training data first");
//		return {}; // return an empty vector since the model hasn't been fitted yet
//	}
//
//    //convert testdata to matrix representation
//    eigen::matrixxd x(testdata.size(), testdata[0].size());
//    for (int i = 0; i < testdata.size(); i++) {
//        for (int j = 0; j < testdata[0].size(); j++)
//            //construct the design matrix x
//            x(i, j) = testdata[i][j];
//    }
//
//    //make predictions using the stored coefficients
//    eigen::matrixxd y = x * m_coefficients;
//    //convert predictions to a vector
//    std::vector<double> result;
//    for (int i = 0; i < y.rows(); i++) {
//        result.push_back(y(i, 0));
//    }
//    
//     // this implementation is using matrix form method    
//    /* implement the following
//		--- check if the model has been fitted
//		--- convert testdata to matrix representation
//		--- construct the design matrix x
//		--- make predictions using the stored coefficients
//		--- convert predictions to a vector
//	*/
//	
//	// todo
//
//
//    return result;
//}



// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {
    
    //Check if training data and labels are of the same size
    if (trainData.size() == trainLabels.size()) {

        //Convert trainData to matrix representation
        Eigen::MatrixXd X(trainData.size(), trainData[0].size());
        for (int i = 0; i < trainData.size(); i++) {
            for (int j = 0; j < trainData[0].size(); j++)
                //Construct the design matrix X{
				X(i, j) = trainData[i][j];
			}
		//convert trainLabels to matrix representation
        Eigen::MatrixXd Y(trainLabels.size(), 1);
        for (int i = 0; i < trainLabels.size(); i++) {
			Y(i, 0) = trainLabels[i];
		}

        //Calculate the coefficients using the least squares method and store them for future predictions
        m_coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * Y);

    }
    else {
		MessageBox::Show("Training data and labels are not of the same size");
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
}



// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {
    if (m_coefficients.size() == 0) {
		MessageBox::Show("Please fit the model to the training data first");
		return {}; // Return an empty vector since the model hasn't been fitted yet
	}

    //Convert testData to matrix representation
     Eigen::MatrixXd X(testData.size(), testData[0].size());
     for (int i = 0; i < testData.size(); i++) {
     	for (int j = 0; j < testData[0].size(); j++)
     		//Construct the design matrix X
     		X(i, j) = testData[i][j];
     }
     
     //Make predictions using the stored coefficients
     Eigen::MatrixXd Y = X * m_coefficients;
     //Convert predictions to a vector
     std::vector<double> result;
     for (int i = 0; i < Y.rows(); i++) {
     	result.push_back(Y(i, 0));
     }
	
     // This implementation is using Matrix Form method    
    /* Implement the following
		--- Check if the model has been fitted
		--- Convert testData to matrix representation
		--- Construct the design matrix X
		--- Make predictions using the stored coefficients
		--- Convert predictions to a vector
	*/
	
	// TODO


    return result;
}



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
		
        //First fit function call
        fit(trainData, trainLabels);
       
        //fit(trainData, trainLabels, 5000, 0.07);

        //Second predict function call
        //std::vector<double> testPredictions = predict(testData, 0);


        // Make predictions on the test data
        std::vector<double> testPredictions = predict(testData);

        // Calculate evaluation metrics (e.g., MAE, MSE)
        double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
        double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
        double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

        //Second predict function call
        //std::vector<double> trainPredictions = predict(trainData, 0);

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