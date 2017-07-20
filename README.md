# ML-Classification-Sample
A project that utilizes several classification algorithms in maximizing prediction rates:
* K Nearest Neighbors
* Linear Discriminant Analysis
* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Naive Bayes  

The data for this project is taken from the wolfram repository at https://datarepository.wolframcloud.com/resources/Sample-Data-Swiss-Bank-Notes.  

I find that, among the listed algorithms, KNN at 10 neighbors provides the best predictions as to the genuity of the bills.  However, more simply, the diagonal values between counterfeit and genuine bills have a clean linear separation; it is likely sufficient to use the diagonal measurement as the discriminant.

REQUIRED PACKAGES
* Numpy
* Matplotlib
* Scikit-learn
* Seaborn
* Pandas
