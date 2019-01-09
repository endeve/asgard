
#include "matlab_utilities.hpp"
#include "tensors.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// matlab's "linspace(start, end, N)" function
//-----------------------------------------------------------------------------
//
// c++ implementation of matlab (a subset of) linspace() function
// initial c++ implementation by Tyler McDaniel
//
// -- linspace (START, END)
// -- linspace (START, END, N)
//     Return a row vector with N linearly spaced elements between START
//     and END.
//
//     If the number of elements is greater than one, then the endpoints
//     START and END are always included in the range.  If START is
//     greater than END, the elements are stored in decreasing order.  If
//     the number of points is not specified, a value of 100 is used.
//
//     The 'linspace' function returns a row vector when both START and
//     END are scalars.
//
//  (unsupported)
//     If one, or both, inputs are vectors, then
//     'linspace' transforms them to column vectors and returns a matrix
//     where each row is an independent sequence between
//     'START(ROW_N), END(ROW_N)'.
//
//     For compatibility with MATLAB, return the second argument (END)
//     when only a single value (N = 1) is requested.
//
//-----------------------------------------------------------------------------
template<typename P>
std::enable_if_t<std::is_floating_point<P>::value, fk::vector<P>>
linspace(P const start, P const end, unsigned int const num_elems)
{
  assert(num_elems > 1); // must have at least 2 elements

  // create output vector
  fk::vector<P> points(num_elems);

  // find interval size
  P const interval_size = (end - start) / (num_elems - 1);

  // insert first and last elements
  points(0)             = start;
  points(num_elems - 1) = end;

  // fill in the middle
  for (unsigned int i = 1; i < num_elems - 1; ++i)
  {
    points(i) = start + i * interval_size;
  }

  return points;
}
//-----------------------------------------------------------------------------
//
// c++ implementation of (a subset of) eye() function
// The following are not supported here:
// - providing a third "CLASS" argument
// - providing a vector argument for the dimensions
//
// -- eye (N)
// -- eye (M, N)
// -- eye ([M N])
//     Return an identity matrix.
//
//     If invoked with a single scalar argument N, return a square NxN
//     identity matrix.
//
//     If supplied two scalar arguments (M, N), 'eye' takes them to be the
//     number of rows and columns.  If given a vector with two elements,
//     'eye' uses the values of the elements as the number of rows and
//     columns, respectively.  For example:
//
//          eye (3)
//           =>  1  0  0
//               0  1  0
//               0  0  1
//
//     The following expressions all produce the same result:
//
//          eye (2)
//          ==
//          eye (2, 2)
//          ==
//          eye (size ([1, 2; 3, 4]))
//
//     Calling 'eye' with no arguments is equivalent to calling it with an
//     argument of 1.  Any negative dimensions are treated as zero.  These
//     odd definitions are for compatibility with MATLAB.
//
//-----------------------------------------------------------------------------
template<typename P>
fk::matrix<P> eye(int const M)
{
  fk::matrix<P> id(M, M);
  for (auto i = 0; i < M; ++i)
    id(i, i) = 1.0;
  return id;
}
template<typename P>
fk::matrix<P> eye(int const M, int const N)
{
  fk::matrix<P> id(M, N);
  for (auto i = 0; i < (M < N ? M : N); ++i)
    id(i, i) = 1.0;
  return id;
}

//-----------------------------------------------------------------------------
// C++ implementation of subset of matlab polyval
// Function for evaluating a polynomial.
//
// Returns the value of a polynomial p evaluated for
// x / each element of x.
// p is a vector of length n+1 whose elements are
// the coefficients of the polynomial in descending powers.

// y = p(0)*x^n + p(1)*x^(n-1) + ... + p(n-1)*x + p(n)
//-----------------------------------------------------------------------------
template<typename P>
P polyval(fk::vector<P> const p, P const x)
{
  int const num_terms = p.size();
  assert(num_terms > 0);

  P y = static_cast<P>(0.0);
  for (int i = 0; i < num_terms - 1; ++i)
  {
    int const deg = num_terms - i - 1;
    y += p(i) * static_cast<P>(std::pow(x, deg));
  }
  y += p(num_terms - 1);

  return y;
}

template<typename P>
fk::vector<P> polyval(fk::vector<P> const p, fk::vector<P> const x)
{
  int const num_terms = p.size();
  int const num_sols  = x.size();
  assert(num_terms > 0);
  assert(num_sols > 0);

  fk::vector<P> solutions(num_sols);
  for (int i = 0; i < num_sols; ++i)
  {
    solutions(i) = polyval(p, x(i));
  }

  return solutions;
}

//-----------------------------------------------------------------------------
//
// these binary files can be generated from matlab or octave with
//
// function writeToFile(path, toWrite)
// fd = fopen(path,'w');
// fwrite(fd,toWrite,'double');
// fclose(fd);
// end
//
//-----------------------------------------------------------------------------
fk::vector<double> readVectorFromBinFile(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in | std::ios::binary);

  // read failed, return empty
  if (!infile)
  {
    return {};
  }

  std::streampos bytes;

  // Get size, seek back to beginning
  infile.seekg(0, std::ios::end);
  bytes = infile.tellg();
  infile.seekg(0, std::ios::beg);

  // create output vector
  fk::vector<double> values;

  unsigned int const num_values = bytes / sizeof(double);
  values.resize(num_values);

  infile.read(reinterpret_cast<char *>(values.data()), bytes);

  return values;

  // infile implicitly closed on exit
}

//
// these ascii files can be generated in octave with, e.g.,
//
// w = 2
// save outfile.dat w
//
// FIXME unsure what Matlab ascii files look like
//
//-----------------------------------------------------------------------------
double readScalarFromTxtFile(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed, return empty
  if (!infile)
  {
    return {};
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: scalar"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "type:");
  infile >> tmp_str;
  assert(tmp_str == "scalar");

  double value;

  infile >> value;

  return value;
}

//-----------------------------------------------------------------------------
//
// these ascii files can be generated in octave with, e.g.,
//
// w = linspace(-1,1);
// save outfile.dat w
//
// FIXME unsure what Matlab ascii files look like
//
//-----------------------------------------------------------------------------
fk::vector<double> readVectorFromTxtFile(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed, return empty
  if (!infile)
  {
    return {};
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: matrix"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "type:");
  infile >> tmp_str;
  assert(tmp_str == "matrix");

  // get the number of rows
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "rows:");
  infile >> tmp_str;
  int rows = std::stoi(tmp_str);

  // get the number of columns
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "columns:");
  infile >> tmp_str;
  int columns = std::stoi(tmp_str);

  // make sure we're working with a (either row or column) vector
  assert((rows == 1) || columns == 1);

  int const num_elems = (rows >= columns) ? rows : columns;

  // create output vector
  fk::vector<double> values;

  values.resize(num_elems);

  for (auto i = 0; i < num_elems; ++i)
  {
    infile >> values(i);
  }

  return values;
}

//-----------------------------------------------------------------------------
//
// these ascii files can be generated in octave with, e.g.,
//
// m = rand(3,3)
// save outfile.dat m
//
// FIXME unsure what Matlab ascii files look like
//
//-----------------------------------------------------------------------------
fk::matrix<double> readMatrixFromTxtFile(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed, return empty
  if (!infile)
  {
    return {};
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: matrix"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "type:");
  infile >> tmp_str;
  assert(tmp_str == "matrix");

  // get the number of rows
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "rows:");
  infile >> tmp_str;
  int rows = std::stoi(tmp_str);

  // get the number of columns
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "columns:");
  infile >> tmp_str;
  int columns = std::stoi(tmp_str);

  // create output matrix
  fk::matrix<double> values(rows, columns);

  for (auto i = 0; i < rows; ++i)
    for (auto j = 0; j < columns; ++j)
    {
      infile >> tmp_str;
      values(i, j) = std::stod(tmp_str);
    }

  return values;
}

// explicit instantiations
template fk::vector<float> linspace(float const start, float const end,
                                    unsigned int const num_elems = 100);
template fk::vector<double> linspace(double const start, double const end,
                                     unsigned int const num_elems = 100);

template fk::matrix<int> eye(int const M = 1);
template fk::matrix<float> eye(int const M = 1);
template fk::matrix<double> eye(int const M = 1);
template fk::matrix<int> eye(int const M, int const N);
template fk::matrix<float> eye(int const M, int const N);
template fk::matrix<double> eye(int const M, int const N);

template int polyval(fk::vector<int> const p, int const x);
template float polyval(fk::vector<float> const p, float const x);
template double polyval(fk::vector<double> const p, double const x);

template fk::vector<int>
polyval(fk::vector<int> const p, fk::vector<int> const x);
template fk::vector<float>
polyval(fk::vector<float> const p, fk::vector<float> const x);
template fk::vector<double>
polyval(fk::vector<double> const p, fk::vector<double> const x);
