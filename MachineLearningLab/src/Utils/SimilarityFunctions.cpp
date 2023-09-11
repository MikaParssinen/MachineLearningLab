#include "SimilarityFunctions.h"
#include <cmath>
#include <stdexcept>


										// SimilarityFunctions class implementation //
			

/// hammingDistance function: Calculates the Hamming distance between two vectors.
double SimilarityFunctions::hammingDistance(const std::vector<double>& v1, const std::vector<double>& v2) {
	if (v1.size() != v2.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Hamming Distance
	//TODO

	return dist;
}


/// jaccardDistance function: Calculates the Jaccard distance between two vectors.
double SimilarityFunctions::jaccardDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double num = 0.0;
	double den = 0.0;
	double dist = 0.0;
	
	// Compute the Jaccard Distance
	// TODO

	return dist;
}


/// cosineDistance function: Calculates the cosine distance between two vectors.///
double SimilarityFunctions::cosineDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dotProduct = 0.0;
	double normA = 0.0;
	double normB = 0.0;
	double cosinedist = 0.0;	
	
	// Compute the cosine Distance
	// TODO

	
	return cosinedist;
}


//NY FUNKTION 
double SimilarityFunctions::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;

	// Compute the Euclidean Distance
	for (size_t i = 0; i < a.size(); ++i) {
		double diff = a[i] - b[i];
		dist += diff * diff;
	}

	return std::sqrt(dist); // Ta kvadratroten av det ackumulerade avståndet
}

/*double SimilarityFunctions::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;

	// Compute the Euclidean Distance
	// TODO


	return dist;
}
*/


/// manhattanDistance function: Calculates the Manhattan distance between two vectors.///
double SimilarityFunctions::manhattanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	for(size_t i = 0; i < a.size(); i++){
		double diff = abs(a[i] - b[i]);  // Ta absolutbeloppet av differensen mellan a och b
		dist += diff + diff;
	}

	// Compute the Manhattan Distance
	// TODO

	return dist;
}

/// minkowskiDistance function: Calculates the Minkowski distance between two vectors.///
double SimilarityFunctions::minkowskiDistance(const std::vector<double>& a, const std::vector<double>& b, int p) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Minkowski Distance
	// TODO
	

	return dist;
}


