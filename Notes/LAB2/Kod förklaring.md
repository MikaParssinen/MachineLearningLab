### Entropi: 
Vi börjar med att göra koden för att räkna ut entropin, ett krav är att entropin räknas ut med Shannons entropi vilket är ett mått på osäkerhet eller information i ett system. Den räknar ut den genomsnittliga mängden information som behövs för att representera händelser i ett system.

Shannons:
$H(X)=−\sum_{i=1} ^{n}​p(x_i​)⋅log_2​(p(x_i​))$ 

$H(X)$ = Entropi för en slupmässig variabel X
$n$ = Möjliga antal utfall
$p(x_i)$ = Sannolikheten för utfallet $x_i$

$log_2$ = Delar upp bitarna i ja/nej. 
##### Entropi kod:
1. 
``` c++
double EntropyFunctions::entropy(const std::vector<double>& y) {
	int total_samples = y.size();
	std::unordered_map<double, int> label_map;
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
```
2. 
``` C++
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
```




### Information Gain:
