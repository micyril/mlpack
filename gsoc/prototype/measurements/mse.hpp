/**
 * The class MeanSquaredError implements the measurement of performance for
 * regression algorithms that is equal to the mean squared error between
 * predicted values and ground truth (correct) values for given test items.
 */
class MeanSquaredError
{
public:
  /**
   * Run prediction and calculate the mean squared error.
   *
   * @param model A test classification model.
   * @data Column-major data containing test items.
   * @responces Ground truth (correct) target values for the test items, should
   *     be either a vector or a column-major matrix.
   */
  template<typename MLAlgorithm, typename DataType, typename ResponcesType>
  static double Evaluate(MLAlgorithm& model, const DataType& data,
      const ResponcesType& responses);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to minimize the measurement.
   */
  static const bool NeedsMinimization = true;
};

