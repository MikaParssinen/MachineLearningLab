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
std::vector<double> KNNClassifier::predict(const std::vector<std::vector<double>>& X_test)
```
