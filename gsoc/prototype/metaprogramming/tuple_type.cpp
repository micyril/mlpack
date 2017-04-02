#include <tuple>

/* Declaring a template and defining it when CTypes is empty */
template<typename...CTypes>
struct CollectionTypes
{
  template<typename...VTypes>
  struct ValueTypes
  {
    using Tuple = std::tuple<VTypes...>;
  };
};

/* Defining a template when CTypes is not empty */
template<typename Head, typename...Tail>
struct CollectionTypes<Head, Tail...>
{
  template<typename...VTypes>
  struct ValueTypes
  {
    using Tuple = typename CollectionTypes<Tail...>::template
        ValueTypes<VTypes..., typename Head::value_type>::Tuple;
  };

  using TupleOfValues = typename ValueTypes<>::Tuple;
};

template<typename...Collections>
using TupleOfValues = typename CollectionTypes<Collections...>::TupleOfValues;

#include <iostream>
#include <array>
#include <armadillo>

int main()
{
  using int_array = std::array<int, 10>;

  using t1 = TupleOfValues<int_array>;
  using t2 = TupleOfValues<int_array, arma::vec>;
  using t3 = TupleOfValues<int_array, arma::vec, int_array>;

  std::cout << std::is_same<t1, std::tuple<int>>::value
            << std::endl; // prints 1
  std::cout << std::is_same<t2, std::tuple<int, double>>::value
            << std::endl; // prints 1
  std::cout << std::is_same<t3, std::tuple<int, double, int>>::value
            << std::endl; // prints 1
}
