#include <type_traits>

template<template<typename...> class MethodTemplate,
         typename Class,
         int AdditionalArgsCount>
struct MethodTypeDetecter;

template<template<typename...> class MethodTemplate, typename Class>
struct MethodTypeDetecter<MethodTemplate, Class, 0>
{
  void operator()(MethodTemplate<Class>);
};

template<template<typename...> class MethodTemplate, typename Class>
struct MethodTypeDetecter<MethodTemplate, Class, 1>
{
  template<typename T1>
  void operator()(MethodTemplate<Class, T1>);
};

template<template<typename...> class MethodTemplate, typename Class>
struct MethodTypeDetecter<MethodTemplate, Class, 2>
{
  template<typename T1, typename T2>
  void operator()(MethodTemplate<Class, T1, T2>);
};

template<typename>
struct True
{
  const static bool value = true;
};

template<typename FunReturnDecltype, typename Result = void>
using EnableIfCompiles =
    typename std::enable_if<True<FunReturnDecltype>::value, Result>::type;

template<template<typename...> class MethodTemplate,
          typename Class,
          int AdditionalArgsCount>
struct HasTrain
{
  using yes = char[1];
  using no = char[2];

  // Making short aliases
  template<template<typename...> class MT, typename C, int N>
  using MTD = MethodTypeDetecter<MT, C, N>;
  template<typename C, typename...Args>
  using MT = MethodTemplate<C, Args...>;
  static const bool N = AdditionalArgsCount;

  template<typename C>
  static EnableIfCompiles<decltype(MTD<MT, C, N>()(&C::Train)), yes&> chk(int);
  template<typename>
  static no& chk(...);

  static bool const value = sizeof(chk<Class>(0)) == sizeof(yes);
};

#include <iostream>

#include <armadillo>

class A
{
public:
  void Train(const arma::mat&, const arma::Row<size_t>&, int);
  void Train(const arma::vec&, size_t, int);
};

class B
{
public:
  void Train(const arma::mat&, const arma::rowvec&);
};

template<typename Class, typename...T>
using TrainForm1 =
    void(Class::*)(const arma::mat&, const arma::Row<size_t>&, T...);

template<typename Class, typename...T>
using TrainForm2 = void(Class::*)(const arma::mat&, const arma::rowvec&, T...);

int main()
{
  std::cout << HasTrain<TrainForm1, A, 0>::value << std::endl; //prints 0
  std::cout << HasTrain<TrainForm1, A, 1>::value << std::endl; //prints 1
  std::cout << HasTrain<TrainForm1, B, 0>::value << std::endl; //prints 0
  std::cout << HasTrain<TrainForm1, B, 1>::value << std::endl; //prints 0
  std::cout << std::endl;
  std::cout << HasTrain<TrainForm2, A, 0>::value << std::endl; //prints 0
  std::cout << HasTrain<TrainForm2, A, 1>::value << std::endl; //prints 0
  std::cout << HasTrain<TrainForm2, B, 0>::value << std::endl; //prints 1
  std::cout << HasTrain<TrainForm2, B, 1>::value << std::endl; //prints 0
}
