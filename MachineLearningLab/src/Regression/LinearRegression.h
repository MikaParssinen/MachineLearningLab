#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <Eigen/Core>

										/// LinearRegression class definition ///

class LinearRegression {
public:
    double hypo(std::vector<double> trainLabels);
    void fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, int num_iterations, double learning_rate);
    std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData, int gradient);
    std::vector<double> predict(const std::vector<std::vector<double>>& testData);
    std::tuple<double, double, double, double, double, double,
        std::vector<double>, std::vector<double>,
        std::vector<double>, std::vector<double>>
        runLinearRegression(const std::string& filePath, int trainingRatio);


    void fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels);
    std::vector<std::vector<double>> LinearRegression::NormalizeForPredict(const std::vector<std::vector<double>>& testData);
    

    
   
   
private:
    Eigen::VectorXd theta;
    Eigen::VectorXd m_coefficients; // Store the coefficients for future predictions

    Eigen::VectorXd m_mean;  // Mean of each feature
    Eigen::VectorXd m_std;   // Standard deiation of each feature
    Eigen::VectorXd m_coefficients;  // Coefficients of the linear regression model

    // Helper function to calculate mean and standard deviation of each feature
    void calculateMeanAndStd(const std::vector<std::vector<double>>& trainData) {
        int num_feats = trainData[0].size();
        int num_samples = trainData.size();

        m_mean = Eigen::VectorXd(num_feats);
        m_std = Eigen::VectorXd(num_feats);

        for (int i = 0; i < num_feats; ++i) {
            double sum = 0.0;
            for (int j = 0; j < num_samples; ++j) {
                sum += trainData[j][i];
            }
            m_mean(i) = sum / num_samples;

            double sum_squared_diff = 0.0;
            for (int j = 0; j < num_samples; ++j) {
                sum_squared_diff += (trainData[j][i] - m_mean(i)) * (trainData[j][i] - m_mean(i));
            }
            m_std(i) = std::sqrt(sum_squared_diff / num_samples);
        }
    }

};

#endif // LINEARREGRESSION_H
