#include <armadillo>

/**
 * The class SimpleCV splits data into training and validation sets, runs
 * training on the training set and evaluates performance on the validation set.
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
class SimpleCV
{
public:
  /* After deducing the type of predictions making an alias for that */
  // using PredictionsType = ...using metaprogramming...;

  /**
   * A constructor that splits data into training and validation set.
   *
   * @param xs Column-major data to train and validate on.
   * @param ys Predictions for points from the data. Should be a vector (a
   *     column or a row) or a column-major matrix.
   * @param validationSize A proportion (from 0 to 1) of the data used as a
   *     validation set.
   */
  SimpleCV(const MatType& xs,
             const PredictionsType& ys,
             const float validationSize);

  /**
   * A constructor that splits data into training and validation set in the case
   * of weighted learning.
   *
   * @param xs Column-major data to train and validate on.
   * @param ys Predictions for points from the data. Should be a vector (a
   *     column or a row) or a column-major matrix.
   * @param weights Weights for points from the data.
   * @param validationSize A proportion (from 0 to 1) of the data used as a
   *     validation set.
   */
  SimpleCV(const MatType& xs,
             const PredictionsType& ys,
             const arma::rowvec weights,
             const float validationSize);

  /**
   * Train on the training set and assess performance on the validation set by
   * using the class Measurement. The method uses different constructors of the
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

  //! Access the trained model.
  const MLAlgorithm& GetModel() const;

  //! Modify the trained model.
  MLAlgorithm& GetModel();
};
