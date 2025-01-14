#pragma once

#include "asgard_kronmult_matrix.hpp"
#include "batch.hpp"
#include "pde.hpp"
#include "tensors.hpp"

namespace asgard::solver
{
enum class poisson_bc
{
  dirichlet,
  periodic
};

// simple, node-local test version of gmres
template<typename P>
gmres_info<P>
simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
             fk::matrix<P> const &M, int const restart, int const max_iter,
             P const tolerance);

// solves ( I - dt * mat ) * x = b
template<typename P>
gmres_info<P>
simple_gmres_euler(const P dt, kronmult_matrix<P> const &mat, fk::vector<P> &x,
                   fk::vector<P> const &b, int const restart,
                   int const max_iter, P const tolerance);

template<typename P>
void setup_poisson(const int N_nodes, P const x_min, P const x_max,
                   fk::vector<P> &diag, fk::vector<P> &off_diag);

template<typename P>
void poisson_solver(fk::vector<P> const &source, fk::vector<P> const &A_D,
                    fk::vector<P> const &A_E, fk::vector<P> &phi,
                    fk::vector<P> &E, int const degree, int const N_elements,
                    P const x_min, P const x_max, P const phi_min,
                    P const phi_max, poisson_bc const bc);

} // namespace asgard::solver
