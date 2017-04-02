#include <mlpack/core/data/dataset_mapper.hpp>
#include <mlpack/methods/decision_stump/decision_stump.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>

#include "grid_search_optimizer.hpp"
#include "../validation/kfold_cv.hpp"
#include "../validation/simple_cv.hpp"
#include "../measurements/accuracy.hpp"

using namespace mlpack::regression;
using namespace mlpack::decision_stump;
using namespace mlpack::tree;

void usage()
{
  arma::mat data /* = ... */;
  arma::Row<size_t> labels /* = ... */;

  /* Without weights */
  GridSearchOptimizer<SoftmaxRegression<>, Accuracy, KFoldCV>
      softmaxOptimizer(data, labels, 5);
  std::array<size_t, 1> numClasses = {5}
  arma::vec lambdas = arma::logspace(-3, 1); // {0.001, 0.01, 0.1, 1}
  std::tuple<size_t, double> bestSoftmaxParams =
      softmax_optimizer.Optimize(numClasses, lambdas);
  double bestSoftmaxAccuracy = softmaxOptimizer.BestMeasurement();
  SoftmaxRegression<>& bestSoftmaxModel = softmaxOptimizer.BestModel();

  /* With weights */
  arma::rowvec weights /* = ... */;
  GridSearchOptimizer<DecisionStump<>, Accuracy, KFoldCV, true>
      dStumpOptimizer(data, labels, weights);
  std::array<size_t, 3> bucketSizes = {5, 7, 9};
  std::tuple<size_t, size_t> bestDStumpParams =
      dStumpOptimizer.Optimize(numClasses, bucketSizes);
  double bestDStumpAccuracy = dStumpOptimizer.BestMeasurement();
  DecisionStump<>& bestDStumpModel = dStumpOptimizer.BestModel();

  /* More complex example */
  GridSearchOptimizer<HoeffdingTree<>, Accuracy, SimpleCV>
      hoeffdingTreeOptimizer(data, labels, 0.2);

  // Setting a set of values for each parameter
  std::array<data::DatasetInfo, 1> datasetInfo /* = {...} */;
  std::array<bool, 1> batchTraining = {false};
  arma::vec successProbabilities = arma::regspace(0.9, 0.01, 0.99);
  std::array<size_t, 3> maxSamplesSet = {0, 3};
  std::array<size_t, 3> checkIntervals = {80, 100, 120};
  std::array<size_t, 3> minSamplesSet = {50, 100, 150};

  // Making variables for best parameters
  data::datasetInfo _;
  size_t __;
  bool ___;
  double successProbability;
  size_t maxSamples;
  size_t checkInterval;
  size_t minSamples;

  // Finding best parameters
  auto bestParameters =
      hoeffdingTreeOptimizer.Optimize(datasetInfo, numClasses, batchTraining,
          successProbabilities, maxSamplesSet, checkIntervals, minSamplesSet);

  // Unpacking best parameters
  std::tie(_, __, ___, successProbability, maxSamples, checkInterval,
      minSamples) = bestParameters;
}
