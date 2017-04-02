#include <armadillo>

/**
 * The class GridSearchOptimizer exhaustively searchs over specified values for
 * hyper parameters of a given machine learning algorithm in order to find
 * hyper parameters that lead to the best performance defined by the class
 * Measurement.
 *
 * @tparam MLAlgorithm A regression or classification algorithm.
 * @tparam Measurement A measurement to assess the quality of the trained model.
 * @tparam CV A type of cross-validation used to assess a set of hyper
 *     parameters.
 * @tparam MatType A matrix type of data.
 * @tparam PredictionsType A type of predictions of the algorithm. Usually it is
 *     arma::Row<size_t> for classification algorithms and arma::vec/arma::mat
 *     for regression algorithms.
 * @tparam WeightedLearning An indicator whether weighted learning should be
 *     performed.
 */
template<typename MLAlgorithm,
         typename Measurement,
         template<typename, typename, bool, typename> class CV,
         bool WeightedLearning = false,
         typename MatType = arma::mat>
class GridSearchOptimizer
{
public:
  /* After deducing the type of predictions making an alias for that */
  // using PredictionsType = ...using metaprogramming...;

  /**
   * A constructor that prepares cross-validation for usage.
   *
   * @param xs Column-major data to train and validate on.
   * @param ys Predictions for points from the data.
   * @param cvArgs A pack of other constructor arguments for CV in addition to
   *     data and predictions.
   */
  template<typename...CVArgs>
  GridSearchOptimizer(const MatType& xs,
                      const PredictionsType& ys,
                      const CVArgs& ...cvArgs);

  /**
   * A constructor that prepares cross-validation for usage in the case of
   *     weighted learning.
   *
   * @param xs Column-major data to train and validate on.
   * @param ys Predictions for points from the data.
   * @param weights Weights for points from the data.
   * @param cvArgs A pack of other constructor arguments for CV in addition to
   *     data and predictions.
   */
  template<typename...CVArgs>
  GridSearchOptimizer(const MatType& xs,
                      const PredictionsType& ys,
                      const arma::rowvec weights,
                      const CVArgs& ...cvArgs);

  /**
   * Exhaustively search over specified values for hyper parameters to find ones
   * that lead to the best performance. The method returns an std::tuple of the
   * best parameters.
   *
   * @param parameterCollections A pack of iterable collections, one per
   *     additional constructor agrument in MLAlgorithm. If MLAlgorithm has a
   *     constructor MLAlgorithm(xs, ys, hyperParameter1, ..., hyperParamterN)
   *     or MLAlgorithm(xs, ys, weights, hyperParameter1, ..., hyperParamterN),
   *     then you should call Optimize(valuesForHyperParameter1, ...,
   *     valuesForHyperParameterN).
   */
  template<typename...Collections>
  TupleOfValues<Collections...> Optimize(
      const Collections& ...parameterCollections);

  //! Access the best model.
  const MLAlgorithm& BestModel() const;

  //! Modify the best model.
  MLAlgorithm& BestModel();

  //! Access the performance measurement of the best model.
  double BestMeasurement() const;
};


