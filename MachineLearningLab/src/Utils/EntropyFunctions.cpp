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



/// Calculates the entropy of a given set of labels "y".///
double EntropyFunctions::entropy(const std::vector<double>& y) {
	int total_samples = y.size();
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	double entropy = 0.0;
	
	// Convert labels to unique integers and count their occurrences
	//TODO

	for (const double& label : y) //Loopar igenom varje label i vector y
	{
		if (label_map.find(label) == label_map.end())//Kollar om labeln redan finns i label_map, om denna if-sats är sann så finns den ej då label_map.find(label) returnerar exakt label_map.end() om den inte finns.
		{
			hist.push_back(label);//vi lägger till den unika labeln i hist som innehåller alla unika labels
			label_map[label] = 1;//vi sätter även första förekomsten av denna label i label_map till 1 då det är den första förekomsten!
		}
		else
		{
			label_map[label]++;//Om den redan finns så betyder det att vi bara ska öka antalet förekomster av labeln, det betyder även att den finns i hist!
		}
	}
	
	
	// Compute the probability and entropy
	//TODO

	 

	for (const double& label : hist)
	{
		int label_count = label_map[label];
		double probability = static_cast<double> (label_count) / total_samples;
		entropy -= probability * log2(probability);
	}

	return entropy;
}


/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	int total_samples = idxs.size();
	double entropy = 0.0;
	// Convert labels to unique integers and count their occurrences
	//TODO

	for (const int& place_to_look_for : idxs)
	{
		if (label_map.find(place_to_look_for) == label_map.end())
		{
			hist.push_back(place_to_look_for);
			label_map[place_to_look_for] = 1;
		}
		else
		{
			label_map[place_to_look_for]++;
		}
	}

	// Compute the probability and entropy
	//TODO

	for (const double& label : hist)
	{
		int label_count = label_map[label];
		double probability = static_cast<double> (label_count) / total_samples;
		entropy -= probability * log2(probability);
	}


	


	return entropy;
}


