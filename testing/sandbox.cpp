#include "batch.hpp"

#include "build_info.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "tools.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>

#ifdef ASGARD_USE_DOUBLE_PREC
using prec = double;
#else
using prec = float;
#endif

#include <Accelerate/Accelerate.h>

using namespace asgard;

void poisson_solver( fk::vector<prec> const &source, fk::vector<prec> &phi, fk::vector<prec> &E, int degree, int N_elements, double x_min, double x_max, double phi_min, double phi_max )
{
    
    // Solving: - phi_xx = source Using Linear Finite Elements
    // Boundary Cconditions: phi(x_min)=phi_min and phi(x_max)=phi_max
    // Returns phi and E = - Phi_x in Gauss-Legendre Nodes
    
    double dx = ( x_max - x_min ) / (double) N_elements;
    
    auto const lgwt = legendre_weights( degree+1, -1.0, 1.0, true );
    
    int N_nodes = N_elements - 1;
    
    // Set the Source Vector //
    
    fk :: vector<prec> b(N_nodes);
    
    for ( int i = 0; i < N_nodes; i++ )
    {
        b[i] = 0.0;
        for ( int q = 0; q < degree+1; q++ )
        {
            b[i] += 0.25 * dx * lgwt[1][q]
                    * (   source[(i  )*(degree+1)+q] * ( 1.0 + lgwt[0][q] )
                        + source[(i+1)*(degree+1)+q] * ( 1.0 - lgwt[0][q] ) );
        }
    }
    
    // Set the Matrix //
    
    // Diagonal Entries //
    
    fk :: vector<prec> A_D(N_nodes);
    
    for ( int i = 0; i < N_nodes; i++ )
    {
        A_D[i] = 2.0 / dx;
    }
    
    // Off-Diagonal Entries //
    
    fk :: vector<prec> A_E(N_nodes-1);
    
    for ( int i = 0; i < N_nodes-1; i++ )
    {
        A_E[i] = - 1.0 / dx;
    }
    
    // Matrix Factorization //
    
    int INFO = 0;
    
    dpttrf_( &N_nodes, A_D.data(), A_E.data(), &INFO );
    
    // Linear Solve //
    
    int NRHS = 1;
    
    dpttrs_( &N_nodes, &NRHS, A_D.data(), A_E.data(), b.data(), &N_nodes, &INFO );
    
    // Set Potential and Electric Field in DG Nodes //
    
    double dg = ( phi_max - phi_min ) / ( x_max - x_min );
    
    // First Element //
    
    for ( int k = 0; k < degree+1; k++ )
    {
        
        double x_k = x_min + 0.5 * dx * ( 1.0 + lgwt[0][k] );
        double g_k = phi_min + dg * ( x_k - x_min );
        
        phi[k] = 0.5 * b[0] * ( 1.0 + lgwt[0][k] ) + g_k;
        
        E[k] = - b[0] / dx - dg;
        
    }
    
    // Interior Elements //
    
    for ( int i = 1; i < N_elements-1; i++ )
    {
        for ( int q = 0; q < degree+1; q++ )
        {
            
            int k      = i * (degree+1) + q;
            double x_k = ( x_min + i * dx ) + 0.5 * dx * ( 1.0 + lgwt[0][q] );
            double g_k = phi_min + dg * ( x_k - x_min );
            
            phi[k] = 0.5 * (   b[i-1] * ( 1.0 - lgwt[0][q] )
                             + b[i]   * ( 1.0 + lgwt[0][q] ) ) + g_k;
            
            E[k] = - ( b[i] - b[i-1] ) / dx - dg;
            
        }
    }
    
    // Last Element //
    
    int i = N_elements-1;
    
    for ( int q = 0; q < degree+1; q++ )
    {
        
        int k      = i * (degree+1) + q;
        double x_k = ( x_min + i * dx ) + 0.5 * dx * ( 1.0 + lgwt[0][q] );
        double g_k = phi_min + dg * ( x_k - x_min );
        
        phi[k] = 0.5 * b[i-1] * ( 1.0 - lgwt[0][q] ) + g_k;
        
        E[k] = b[i-1] / dx - dg;
        
    }
    
}

int main(int, char**)
{
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
    
    int    const N_elements = 16;
    int    const N_nodes    = N_elements + 1;
    int    const degree     = 2;
    double const x_min = 0.0;
    double const x_max = 1.0;
    double const phi_min = 0.0;
    double const phi_max = 0.0;
    fk :: vector<prec> poisson_source((degree+1)*N_elements);
    fk :: vector<prec> poisson_phi   ((degree+1)*N_elements);
    fk :: vector<prec> poisson_E     ((degree+1)*N_elements);
    fk :: vector<prec> x             ((degree+1)*N_elements);
    fk :: vector<prec> x_e(N_nodes);
    
    // Assume Uniform Elements //
    
    double dx = ( x_max - x_min ) / (double) N_elements;
    
    // Set Finite Element Nodes //
    
    for ( int i = 0; i < N_nodes; i++ )
    {
        
        x_e[i] = x_min + i * dx;
        
    }
    
    // Set Source in DG Elements //
    
    auto const lgwt = legendre_weights( degree+1, -1.0, 1.0, true );
    
    for ( int i = 0; i < N_elements; i++ )
    {
        for ( int q = 0; q < degree+1; q++ )
        {
            
            int k = i*(degree+1)+q;
            
            double x_q = lgwt[0][q];
            
            x[k] = x_e[i] + 0.5 * dx * ( 1.0 + x_q );
            
            poisson_source[k] = sin( 2.0 * M_PI * x[k] );
            
        }
    }
    
    poisson_solver
    ( poisson_source, poisson_phi, poisson_E, degree, N_elements,
      x_min, x_max, phi_min, phi_max );
    
    std::cout << "---- x ----\n";
    for ( int i = 0; i < N_elements; i++ )
    {
        for ( int q = 0; q < degree+1; q++ )
        {
            
            int k = i*(degree+1)+q;
            
            std::cout << x[k] << "\n";
            
        }
        
    }
    
    double error = 0.0;
    
    std::cout << "--- phi ---\n";
    for ( int i = 0; i < N_elements; i++ )
    {
        for ( int q = 0; q < degree+1; q++ )
        {
            
            int k = i*(degree+1)+q;
            
            std::cout << poisson_phi[k] << "\n";
            
            error += pow(poisson_phi[k]-sin(2.0*M_PI*x[k])/pow(2.0*M_PI,2),2);
            
        }
        
    }
    std::cout << "-----------\n";
    
    error = sqrt( error ) / ( (degree+1)*N_elements );
    
    std::cout << "Error = " << error << "\n";
    
  return 0;
}
