#include "EntropyFunctions.h"
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>


									// EntropyFunctions class implementation //






double EntropyFunctions::entropy(const std::vector<double>& y) {
    int total_samples = y.size();
    std::unordered_map<double, int> label_map;
	//We do not use hist we save it in an hashmap.


    // Count occurrences of each label
    for (double label : y) {
        label_map[label]++;
    }

    double entropy = 0.0;

    // Compute the probability and entropy
    for (const auto& entry : label_map) {
        double probability = static_cast<double>(entry.second) / total_samples;
        entropy -= probability * log2(probability);
    }

    return entropy;
}


/// Calculates the entropy of a given set of labels "y".///
//double EntropyFunctions::entropy(const std::vector<double>& y) {
	//int total_samples = y.size();
	//std::vector<double> hist;
	//std::unordered_map<double, int> label_map;
	//double entropy = 0.0;
	
	// Convert labels to unique integers and count their occurrences
	//TODO
	
	// Compute the probability and entropy
	//TODO

	//return entropy;
//}


/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	int total_samples = idxs.size();
	double entropy = 0.0;
	// Convert labels to unique integers and count their occurrences
	//TODO


	// Compute the probability and entropy
	//TODO


	return entropy;
}


