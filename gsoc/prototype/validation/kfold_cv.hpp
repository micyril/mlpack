#include <armadillo>

/**
 * The class KFoldCV implements k-fold cross validation for regression and
 * classification algorithms.
 *
 * @tparam MLAlgorithm A regression or classification algorithm.
 * @tparam Measurement A measurement to assess the quality of the trained model.
 * @tparam MatType A matrix type of data.
 * @tparam PredictionsType A type of predictions of the algorithm. Usually it is
 *     arma::Row<size_t> for classification algorithms and arma::vec/arma::mat
 *     for regression algorithms.
 * @tparam WeightedLearning An indicator whether weighted learning should be
 *     performed.
 */
template<typename MLAlgorithm,
         typename Measurement,
         bool WeightedLearning = false,
         typename MatType = arma::mat>
class KFoldCV
{
public:
  /* After deducing the type of predictions making an alias for that */
  // using PredictionsType = ...using metaprogramming...;

  /**
   * A constructor that prepares data for performing k-fold cross validation.
   *
   * @param xs Column-major data to train and validate on.
   * @param ys Predictions for points from the data. Should be a vector (a
   *     column or a row) or a column-major matrix.
   * @param k An amount of folds for k-fold cross validation.
   */
  KFoldCV(const MatType& xs, const PredictionsType& ys, const size_t k = 10);

  /**
   * A constructor that prepares data for performing k-fold cross validation
   * with weighted learning.
   *
   * @param xs Column-major data to train and validate on.
   * @param ys Predictions for points from the data. Should be a vector (a
   *     column or a row) or a column-major matrix.
   * @param weights Weights for points from the data.
   * @param k An amount of folds for k-fold cross validation.
   */
  KFoldCV(const MatType& xs,
          const PredictionsType& ys,
          const arma::rowvec weights,
          const size_t k = 10);

  /**
   * Perform k-fold cross validation and return the mean value of measurements
   * for all splits of the data. The method uses different constructors of the
   * class MLAlgorithm depending on the flag WeightedLearning:
   * MLAlgorithm(trainingXs, trainingYs, mlAlgorithmArgs...) when
   * WeightedLearning is false, and MLAlgorithm(trainingXs, trainingYs,
   * trainingWeights, mlAlgorithmArgs...) otherwise.
   *
   * @param mlAlgorithmArgs A pack of other constructor arguments for
   *     MLAlgorithm in addition to data and predictions (and weights in the
   *     case of weighted learning).
   */
  template<typename...MLAlgorithmArgs>
  double Evaluate(const MLAlgorithmArgs& ...mlAlgorithmArgs);

  //! Access a model trained on one of the splits.
  const MLAlgorithm& GetModel() const;

  //! Modify the model trained on one of the splits.
  MLAlgorithm& GetModel();
};
