##### Results Euclidean
![[Skärmbild (8).png]]
What we see is that with Euclidean we have 97% Accuracy on the **Training Set**, while on the **Test Set** we have 100%. With K=5.
##### Results Manhattan:
![[Skärmbild (9).png]] 
What we see is that with Manhattan we have 96% Accuracy on the **Training Set**, while on the **Test Set** we have 97%. With K=5.



##### Classification
Iris data set has 150 individual data. 
**80/20 split**

| K          | 1   | 12 | 50  | 100 |
| ------------ | --- | --------- | --- | --- |
| Training Acc |   100%  |    99%       |  92%   |  67%   |
| Test Acc             |  97%   |     100%      |   100%  |  54%   |

**With 70/30** 

| K          | 1   | 12 | 50  | 100 |
| ------------ | --- | --------- | --- | --- |
| Training Acc |    100% |       99%    |    93% |   48%  |
| Test Acc             | 96%    |     96%      |   94% | 38%     |

So what dose this mean? With K as 1 we actually have a really good accuracy, i think it is because we are working with a small dataset. 
With K as 12 which is sqrt(150) we have almost perfect accuracy why we choose 12 is because i saw it on the internet that should be de most optimal K. 
K as 50 was a small hit on accuracy not so much that we were hoping on tho..
And finally K as 100 we see that the accuracy is really bad. 

Conclusion: We should choose a k which is not too small, so we avoid overfitting*. Also we should avoid large K so we avoid underfitting* 
Smallest K we could choose was 1 and largest was like 119.


##### Regression

 **Evaluation**

As you can see we have tested using a lot of different values of K. The best one in our opinion is K = 23. This is our chosen value of K since we are using the so called "optimal K value", in which K = sqrt(N), where "N" is the number of samples we have.
sprt(506) is closer to 22 than to 23. However, we tested both K = 22 and K = 23 and K = 23 came out on top with better results on the evaluation functions "RMSE" & "MAE". The value of "RMSE" is commonly supposed to be close to 0. But in this case we are in a span of 0$ - 10s of thousands of $, which proves that RMSE = 5 is a good value. Same for MAE & R-Squared. 

We can see that the datapoints are very close to eachother when K=22 & K = 23. The other K - values show a wider spread of data points which (to our knowledge) makes the prediction worse in some cases. However, when K = 1 the error function results, display somewhat similar results as when K = 22. For example: 

Smallest and biggest K values:

For example when k = 1 there is a lot of widespread points at the upper right of the graph. Another example is when K=400. Here we can quite easily calculate that the prediction is bad since we are working with around 500 houses and using a K - value that is almost as great as the number of houses we are working with. This leads to brushing every house under one comb which leads to sed horisontal line when K=400 & K=250.  



  





K = 1![[Skärmbild (11).png]]
K = 22![[Skärmbild (13).png]]
K = 23
![[Pasted image 20230913191013.png]]
K = 250![[Skärmbild (15).png]]
K = 400
![[Skärmbild (17).png]]

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

1. Funktionen `predict` tar en 2D vektor `X_test` som indata, vilket representerar testdatapunkterna, i vårat fall är det irisar.
2. En tom vektor `y_pred` skapas för att lagra de förutsagda etiketterna* för varje testdatapunkt.
3. Kontrollerar om träningsdata `X_train_` eller `y_train_` är tom. Om så är fallet, kastas ett undantag.
4. För varje testdatapunkt i `X_test`, beräknas avståndet till alla träningsdatapunkter med hjälp av Euklidiskt avstånd (eller Manhattan avstånd, beroende på vilken rad som är kommenterad ut).
5. Avstånden sorteras i stigande ordning.
6. De närmaste K grannarna väljs utifrån de sorterade avstånden.
7. För dessa K grannar, räknas antalet förekomster av varje etikett.
8. Den etikett som förekommer mest bland de K grannarna väljs som den förutsagda etiketten för den aktuella testdatapunkten.
9. Den förutsagda etiketten läggs till i `y_pred`.
10. När alla testdatapunkter har bearbetats, returneras `y_pred` som innehåller de förutsagda etiketterna för alla testdatapunkter.

Observera att denna kod antar att `X_train_`, `y_train_`, och `k_` är medlemsvariabler i klassen `KNNClassifier`, och att de har satts någonstans tidigare i koden (troligen i en träningsfunktion som inte visas här)



