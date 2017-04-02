#include <mlpack/methods/decision_stump/decision_stump.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

#include "../validation/kfold_cv.hpp"
#include "../validation/simple_cv.hpp"
#include "../measurements/accuracy.hpp"
#include "../measurements/mse.hpp"

using namespace mlpack::regression;
using namespace mlpack::decision_stump;

int main()
{
  arma::mat data /* = ... */;
  arma::Row<size_t> labels /* = ... */;
  size_t numClasses = 5;

  /* Without weights */
  KFoldCV<SoftmaxRegression<>, Accuracy> softmaxCV(data, labels);
  double lambda = 0.1;
  double softmaxAccuracy = softmaxCV.Evaluate(numClasses, lambda);

  /* With weights */
  arma::rowvec weights /* = ... */;
  KFoldCV<DecisionStump<>, Accuracy, true> dStumpCV(data, labels, weights);
  size_t bucketSize = 5;
  double dStumpAccuracy = dStumpCV.Evaluate(numClasses, bucketSize);

  /* Simple validation */
  arma::vec responces /* = ... */;
  float validationPortion = 0.2F;
  SimpleCV<LinearRegression, MeanSquaredError>
      lRegressionSV(data, responces, validationPortion);
  double lRegressionLambda = 0.05;
  double lRegressionMSE = lRegressionSV.Evaluate(lRegressionLambda);
}
