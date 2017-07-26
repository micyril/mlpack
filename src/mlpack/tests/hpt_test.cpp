/**
 * @file hpt_test.cpp
 *
 * Tests for the hyper-parameter tuning module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core/cv/metrics/mse.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/hpt/bind.hpp>
#include <mlpack/core/hpt/cv_function.hpp>
#include <mlpack/core/hpt/hpt.hpp>
#include <mlpack/core/optimizers/grid_search/grid_search.hpp>
#include <mlpack/methods/lars/lars.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack::cv;
using namespace mlpack::hpt;
using namespace mlpack::optimization;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(HPTTest);

/**
 * Test CVFunction runs cross-validation in according with specified bound
 * arguments and passed parameters.
 */
BOOST_AUTO_TEST_CASE(CVFunctionTest)
{
  arma::mat xs = arma::randn(5, 100);
  arma::vec beta = arma::randn(5, 1);
  arma::mat ys = beta.t() * xs + 0.1 * arma::randn(5, 1);

  SimpleCV<LARS, MSE> cv(0.2, xs, ys);

  bool transposeData = true;
  bool useCholesky = false;
  double lambda1 = 1.0;
  double lambda2 = 2.0;

  BoundArg<bool, 1> boundUseCholesky{useCholesky};
  BoundArg<double, 3> boundLambda1{lambda2};
  CVFunction<decltype(cv), 4, BoundArg<bool, 1>, BoundArg<double, 3>>
      cvFun(cv, boundUseCholesky, boundLambda1);

  double expected = cv.Evaluate(transposeData, useCholesky, lambda1, lambda2);
  double actual = cvFun.Evaluate(arma::vec{double(transposeData), lambda1});

  BOOST_REQUIRE_CLOSE(expected, actual, 1e-5);
}

void InitProneToOverfittingData(arma::mat& xs,
                                arma::rowvec& ys,
                                double& validationSize)
{
  // Total number of data points.
  size_t N = 10;
  // Total number of features (all except the first one are redundant).
  size_t M = 5;

  arma::rowvec data = arma::linspace<arma::rowvec>(0.0, 10.0, N);
  xs = data;
  for (size_t i = 2; i <= M; ++i)
    xs = arma::join_cols(xs, arma::pow(data, i));

  // Responses that approximately follow the function y = 2 * x. Adding noise to
  // avoid having a polynomial of degree 1 that exactly fits the points.
  ys = 2 * data + 0.05 * arma::randn(1, N);

  validationSize = 0.3;
}

template<typename T1, typename T2>
void FindLARSBestLambdas(arma::mat& xs,
                         arma::rowvec& ys,
                         double& validationSize,
                         bool transposeData,
                         bool useCholesky,
                         const T1& lambda1Set,
                         const T2& lambda2Set,
                         double& bestLambda1,
                         double& bestLambda2,
                         double& bestObjective)
{
  SimpleCV<LARS, MSE> cv(validationSize, xs, ys);

  bestObjective = std::numeric_limits<double>::max();

  for (double lambda1 : lambda1Set)
    for (double lambda2 : lambda2Set)
    {
      double objective =
          cv.Evaluate(transposeData, useCholesky, lambda1, lambda2);
      if (objective < bestObjective)
      {
        bestObjective = objective;
        bestLambda1 = lambda1;
        bestLambda2 = lambda2;
      }
    }
}

 /**
 * Test grid-search optimization leads to the best parameters from the specified
 * ones.
 */
BOOST_AUTO_TEST_CASE(GridSearchTest)
{
  arma::mat xs;
  arma::rowvec ys;
  double validationSize;
  InitProneToOverfittingData(xs, ys, validationSize);

  bool transposeData = true;
  bool useCholesky = false;
  arma::vec lambda1Set =
      arma::join_rows(arma::vec{0}, arma::logspace<arma::vec>(-3, 2, 6));
  std::array<double, 4> lambda2Set{{0.0, 0.05, 0.5, 5.0}};

  double expectedLambda1, expectedLambda2, expectedObjective;
  FindLARSBestLambdas(xs, ys, validationSize, transposeData, useCholesky,
      lambda1Set, lambda2Set, expectedLambda1, expectedLambda2,
      expectedObjective);

  SimpleCV<LARS, MSE> cv(validationSize, xs, ys);

  GridSearch optimizer(lambda1Set, lambda2Set);
  CVFunction<decltype(cv), 4, BoundArg<bool, 0>, BoundArg<bool, 1>>
      cvFun(cv, {transposeData}, {useCholesky});
  arma::mat actualParameters;
  double actualObjective = optimizer.Optimize(cvFun, actualParameters);

  BOOST_REQUIRE_CLOSE(expectedObjective, actualObjective, 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda1, actualParameters(0, 0), 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda2, actualParameters(1, 0), 1e-5);
}

/**
 * Test HyperParameterTuner.
 */
BOOST_AUTO_TEST_CASE(HPTTest)
{
  arma::mat xs;
  arma::rowvec ys;
  double validationSize;
  InitProneToOverfittingData(xs, ys, validationSize);

  bool transposeData = true;
  bool useCholesky = false;
  arma::vec lambda1Set =
      arma::join_rows(arma::vec{0}, arma::logspace<arma::vec>(-3, 2, 6));
  arma::vec lambda2Set{0.0, 0.05, 0.5, 5.0};

  double expectedLambda1, expectedLambda2, expectedObjective;
  FindLARSBestLambdas(xs, ys, validationSize, transposeData, useCholesky,
      lambda1Set, lambda2Set, expectedLambda1, expectedLambda2,
      expectedObjective);

  double actualLambda1, actualLambda2;
  HyperParameterTuner<LARS, MSE, SimpleCV, GridSearch>
      hpt(validationSize, xs, ys);
  std::tie(actualLambda1, actualLambda2) = hpt.Optimize(Bind(transposeData),
      Bind(useCholesky), lambda1Set, lambda2Set);

  BOOST_REQUIRE_CLOSE(expectedObjective, hpt.BestObjective(), 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda1, actualLambda1, 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda2, actualLambda2, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
