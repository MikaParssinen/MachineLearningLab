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
	std::unordered_map<double, int> label_map; //(label, count)
	double entropy = 0.0;

	// Count occurrences of each label
	for (const double& label : y) {
		label_map[label]++;
	}

	// Compute the probability and entropy
	for (const auto& pair : label_map) {
		double probability = static_cast<double>(pair.second) / total_samples;
		entropy -= probability * log2(probability);
	}

	return entropy;
}

double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
	std::unordered_map<double, int> label_map;
	int total_samples = idxs.size();
	double entropy = 0.0;

	// Count occurrences of each label based on indices
	for (const int& idx : idxs) {
		double label = y[idx];
		label_map[label]++;
	}

	// Compute the probability and entropy
	for (const auto& pair : label_map) {
		double probability = static_cast<double>(pair.second) / total_samples;
		entropy -= probability * log2(probability);
	}

	return entropy;
}



