##### Results Euclidean
![[Skärmbild (8).png]]
What we see is that with Euclidean we have 97% Accuracy on the **Training Set**, while on the **Test Set** we have 100%. With K=3.
##### Results Manhattan:
![[Skärmbild (9).png]] 
What we see is that with Manhattan we have 96% Accuracy on the **Training Set**, while on the **Test Set** we have 97%. With K=3.



##### Value of K?
First off we tried K = 3 which means that we compare the 3 nearest neighbors. It gave us 97% and 100%, with other K's we are having the same results. 

##### Code explanation
```cpp
std::vector<double> KNNClassifier::predict(const std::vector<std::vector<double>>& X_test) const {
    std::vector<double> y_pred; // Lagra förutsagda etiketter för alla testdatapunkter
    y_pred.reserve(X_test.size()); // Reservera minne för y_pred för att undvika frekvent omallokering

    // Kontrollera om träningsdata är tom
    if (X_train_.empty() || y_train_.empty()) {
        throw std::runtime_error("Error: Tom träningsdata.");
    }

    // För varje testdatapunkt
    for (const auto& x_test : X_test) {
        // Beräkna avståndet till alla träningsdatapunkter
        std::vector<std::pair<double, double>> distances; // (avstånd, etikett) par
        for (size_t i = 0; i < X_train_.size(); ++i) {
            double distance = SimilarityFunctions::euclideanDistance(x_test, X_train_[i]);
            //double distance = SimilarityFunctions::manhattanDistance(x_test, X_train_[i]);
            distances.emplace_back(distance, y_train_[i]);
        }

        // Sortera avstånden i stigande ordning
        std::sort(distances.begin(), distances.end(),
            [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                return a.first < b.first;
            });

        // Hämta de närmaste K grannarna
        int k = k_; // Använd den K som definierades vid konstruktion
        std::map<double, int> labelCounts; // (etikett, antal) kartläggning
        for (int i = 0; i < k; ++i) {
            labelCounts[distances[i].second]++;
        }

        // Hitta den mest frekventa etiketten
        double predictedLabel = -1; // Standardvärde om inga grannar hittades
        int maxCount = 0;
        for (const auto& pair : labelCounts) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                predictedLabel = pair.first;
            }
        }

        y_pred.push_back(predictedLabel);
    }

    return y_pred; // Returnera vektorn med förutsagda etiketter för alla testdatapunkter
}
```

In Swedish: Ovan är funktionen **PREDICT** som enligt LAB1 skulle göras, kort sagt är det en implementering av #KNN algoritmen för #klassificering.
Detta är steg för steg hur funktionen predict fungerar

1. Funktionen `predict` tar en 2D vektor `X_test` som indata, vilket representerar testdatapunkterna, i detta fall är det irisar.
2. En tom vektor `y_pred` skapas för att lagra de förutsagda *etiketterna* för varje testdatapunkt.
3. Kontrollerar om träningsdata `X_train_` eller `y_train_` är tom. Om så är fallet, kastas ett undantag.
4. För varje testdatapunkt i `X_test`, beräknas avståndet till alla träningsdatapunkter med hjälp av Euklidiskt avstånd (eller Manhattan avstånd, beroende på vilken rad som är kommenterad ut).
5. Avstånden sorteras i stigande ordning.
6. De närmaste K grannarna väljs utifrån de sorterade avstånden.
7. För dessa K grannar, räknas antalet förekomster av varje etikett.
8. Den etikett som förekommer mest bland de K grannarna väljs som den förutsagda etiketten för den aktuella testdatapunkten.
9. Den förutsagda etiketten läggs till i `y_pred`.
10. När alla testdatapunkter har bearbetats, returneras `y_pred` som innehåller de förutsagda etiketterna för alla testdatapunkter.

Observera att denna kod antar att `X_train_`, `y_train_`, och `k_` är medlemsvariabler i klassen `KNNClassifier`, och att de har satts någonstans tidigare i koden (troligen i en träningsfunktion som inte visas här).