#include <armadillo>

/**
 * The class Accuracy implements the classical measurement of performance for
 * classification algorithms that is equal to a proportion of correctly labeled
 * test items among all ones for given test items.
 */
class Accuracy
{
public:
  /**
   * Run classification and calculate accuracy.
   *
   * @param model A test classification model.
   * @data Column-major data containing test items.
   * @labels Ground truth (correct) labels for the test items.
   */
  template<class MLAlgorithm, class DataType>
  static double Evaluate(MLAlgorithm& model, const DataType& data,
      const arma::Row<size_t>& labels);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to maximize the measurement.
   */
  static const bool NeedsMinimization = false;
};
