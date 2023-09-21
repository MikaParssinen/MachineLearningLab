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




### informationGain():

#### GrowTree():
Första uppgiften i GrowTree är att definera ett stopp kriterium, när ska vi sluta växa trädet? Det finns några fall när trädet ska sluta växa: 
1. Antalet fall i noden är mindre än en förutbestämd gräns. Om dett finns för få datapunkter kvar i noden, kan det vara meningslöst att fortsätta dela upp den. 
2. Renheten i noden överstiger en viss gräns. Om nästan alla datapunkter i noden tillhör samma klass, kan det vara onödigt att fortsätta dela upp den. 
3. Djupet av noden överstiger en viss gräns. Detta är ett sätt att förhindra att trädet blir för stort och överanpassat.
4. Prediktorvärderna för alla poster är identiska.

Vad kan våra stopkriterier vara? Jo, vi har ett litet dataset då vill vi ha en en log förut
