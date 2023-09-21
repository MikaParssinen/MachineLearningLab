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


/// Calculates the entropy of a given set of labels "y".///
//double EntropyFunctions::entropy(const std::vector<double>& y) {
//	int total_samples = y.size();
//	std::vector<double> hist;
//	std::unordered_map<double, int> label_map;
//	double entropy = 0.0;
//	// Convert labels to unique integers and count their occurrences
//	//TODO
//	for (const double& label : y) //Loopar igenom varje label i vector y
//	{
//		if (label_map.find(label) == label_map.end())//Kollar om labeln redan finns i label_map, om denna if-sats �r sann s� finns den ej d� label_map.find(label) returnerar exakt label_map.end() om den inte finns.
//		{
//			hist.push_back(label);//vi l�gger till den unika labeln i hist som inneh�ller alla unika labels
//			label_map[label] = 1;//vi s�tter �ven f�rsta f�rekomsten av denna label i label_map till 1 d� det �r den f�rsta f�rekomsten!
//		}
//		else
//		{
//			label_map[label]++;//Om den redan finns s� betyder det att vi bara ska �ka antalet f�rekomster av labeln, det betyder �ven att den finns i hist!
//		}
//	}
//	
//	// Compute the probability and entropy
//	//TODO 
//
//	for (const double& label : hist)
//	{
//		int label_count = label_map[label];
//		double probability = static_cast<double> (label_count) / total_samples;
//		entropy -= probability * log2(probability);
//	}
//
//	return entropy;
//}


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



/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
//double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
//	std::vector<double> hist;
//	std::unordered_map<double, int> label_map;
//	int total_samples = idxs.size();
//	double entropy = 0.0;
//	// Convert labels to unique integers and count their occurrences
//	//TODO
//
//	for (const int& place_to_look_for : idxs)
//	{
//		if (label_map.find(place_to_look_for) == label_map.end())
//		{
//			hist.push_back(place_to_look_for);
//			label_map[place_to_look_for] = 1;
//		}
//		else
//		{
//			label_map[place_to_look_for]++;
//		}
//	}
//
//	// Compute the probability and entropy
//	//TODO
//
//	for (const double& label : hist)
//	{
//		int label_count = label_map[label];
//		double probability = static_cast<double> (label_count) / total_samples;
//		entropy -= probability * log2(probability);
//	}
//
//	return entropy;
//}


