#pragma once
#include "pde_base.hpp"

namespace asgard
{
// 4D collisional landau, i.e.,
//
//  df/dt == -v*\grad_x f -E\grad_v f + div_v( (v-u)f + theta\grad_v f)
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_collisional_landau_1X3V : public PDE<P>
{
public:
  PDE_collisional_landau_1X3V(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_, moments_,
               do_collision_operator_)
  {
    param_manager.add_parameter(parameter<P>{"n", n});
    param_manager.add_parameter(parameter<P>{"u", u});
    param_manager.add_parameter(parameter<P>{"theta", theta});
    param_manager.add_parameter(parameter<P>{"E", E});
    param_manager.add_parameter(parameter<P>{"S", S});
    param_manager.add_parameter(parameter<P>{"MaxAbsE", MaxAbsE});
  }

private:
  static int constexpr num_dims_               = 4;
  static int constexpr num_sources_            = 0;
  static int constexpr num_terms_              = 5;
  static bool constexpr do_poisson_solve_      = true;
  static bool constexpr do_collision_operator_ = true;
  static bool constexpr has_analytic_soln_     = false;
  static int constexpr default_degree          = 3;

  static P constexpr nu       = 1.0;    // collision frequency
  static P constexpr A        = 0.0e-4; // amplitude
  static P constexpr theta_in = 1.0;    // initial temperature

  static fk::vector<P>
  initial_condition_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return 1.0 + A * std::cos(0.5 * x_v);
    });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta_in);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient](P const x_v) -> P {
          return coefficient *
                 std::exp(-0.5 * (1.0 / theta_in) * std::pow(x_v, 2));
        });
    return fx;
  }
  
  static fk::vector<P>
  initial_condition_dim_v_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta_in);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient](P const x_v) -> P {
          return coefficient *
                 std::exp(-0.5 * (1.0 / theta_in) * std::pow(x_v, 2));
        });
    return fx;
  }
  
  static fk::vector<P>
  initial_condition_dim_v_2(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta_in);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient](P const x_v) -> P {
          return coefficient *
                 std::exp(-0.5 * (1.0 / theta_in) * std::pow(x_v, 2));
        });
    return fx;
  }

  static P dV(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  /* Define the dimension */
  
  inline static dimension<P> const dim_0 = dimension<P>(
      -2.0 * PI, 2.0 * PI, 4, default_degree, initial_condition_dim_x_0, dV, "x");

  inline static dimension<P> const dim_1 = dimension<P>(
      -6.0     , 6.0     , 3, default_degree, initial_condition_dim_v_0, dV, "v_1");

  inline static dimension<P> const dim_2 = dimension<P>(
      -6.0     , 6.0     , 3, default_degree, initial_condition_dim_v_1, dV, "v_2");

  inline static dimension<P> const dim_3 = dimension<P>(
      -6.0     , 6.0     , 3, default_degree, initial_condition_dim_v_2, dV, "v_3");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0, dim_1, dim_2, dim_3};

  /* Define the moments */
  
  static fk::vector<P> moment_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> f(x.size());
    std::fill(f.begin(), f.end(), 1.0);
    return f;
  }

  static fk::vector<P> moment_v(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    return fk::vector<P>(x);
  }

  static fk::vector<P> moment_vSqr(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> f(x.size());
    std::transform(x.begin(), x.end(), f.begin(),
                   [](P const &x_v) -> P { return std::pow(x_v, 2); });
    return f;
  }

  inline static moment<P> const moment0 = moment<P>(
      std::vector<md_func_type<P>>({{moment_1, moment_1, moment_1, moment_1, moment_1}}));
  inline static moment<P> const moment1_1 = moment<P>(
      std::vector<md_func_type<P>>({{moment_1, moment_v, moment_1, moment_1, moment_1}}));
  inline static moment<P> const moment1_2 = moment<P>(
      std::vector<md_func_type<P>>({{moment_1, moment_1, moment_v, moment_1, moment_1}}));
  inline static moment<P> const moment1_3 = moment<P>(
      std::vector<md_func_type<P>>({{moment_1, moment_1, moment_1, moment_v, moment_1}}));
  inline static moment<P> const moment2_1 = moment<P>(
      std::vector<md_func_type<P>>({{moment_1, moment_vSqr, moment_1, moment_1, moment_1}}));
  inline static moment<P> const moment2_2 = moment<P>(
      std::vector<md_func_type<P>>({{moment_1, moment_1, moment_vSqr, moment_1, moment_1}}));
  inline static moment<P> const moment2_3 = moment<P>(
      std::vector<md_func_type<P>>({{moment_1, moment_1, moment_1, moment_vSqr, moment_1}}));

  inline static std::vector<moment<P>> const moments_ = {moment0, moment1_1, moment1_2, moment1_3,
                                                         moment2_1, moment2_2, moment2_3};

  /* Construct (n, u, theta) */
  // n = density
  // u = bulk velocity
  // theta = temperature
  static P n(P const &x, P const t = 0)
  {
    ignore(t);

    return (1.0 + A * std::cos(0.5 * x));
  }

  static P u(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  static P theta(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 1.0;
  }

  // E = -d_x phi
  static P E(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  // source for poisson problem is -d_xx phi = n - 1 = S(n)
  static P S(P const &y, P const t = 0)
  {
    ignore(t);
    // subtracts quadrature values by one
    return y - 1.0;
  }

  // holds the maximum absolute value of E
  static P MaxAbsE(P const &x, P const t = 0)
  {
    ignore(t);
    ignore(x);
    return 0.0;
  }

  /* TODO: Make identity term for reuse */

  /* build the terms */

  // Term 1
  // -v\cdot\grad_x f for v > 0
  //
  
  static P e1_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }

  static P e1_g2(P const x, P const time = 0)
  {
    ignore(time);
    return (x > 0.0) ? x : 0.0;
  }
  
  static P e1_g3(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }
  
  static P e1_g4(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  inline static const partial_term<P> e1_pterm_x = partial_term<P>(
      coefficient_type::div, e1_g1, nullptr, flux_type::downwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e1_pterm_v1 = partial_term<P>(
      coefficient_type::mass, e1_g2, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e1_pterm_v2 = partial_term<P>(
      coefficient_type::mass, e1_g3, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e1_pterm_v3 = partial_term<P>(
      coefficient_type::mass, e1_g4, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term_e1x =
      term<P>(false,  // time-dependent
              "E1_x", // name
              {e1_pterm_x}, imex_flag::imex_explicit);

  inline static term<P> const term_e1v1 =
      term<P>(false,  // time-dependent
              "E1_v1", // name
              {e1_pterm_v1}, imex_flag::imex_explicit);

  inline static term<P> const term_e1v2 =
      term<P>(false,  // time-dependent
              "E1_v2", // name
              {e1_pterm_v2}, imex_flag::imex_explicit);

  inline static term<P> const term_e1v3 =
      term<P>(false,  // time-dependent
              "E1_v3", // name
              {e1_pterm_v3}, imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_1 = {term_e1x, term_e1v1, term_e1v2, term_e1v3};

  // Term 2
  // -v\cdot\grad_x f for v < 0
  //
  static P e2_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }

  static P e2_g2(P const x, P const time = 0)
  {
    ignore(time);
    return (x < 0.0) ? x : 0.0;
  }
  
  static P e2_g3(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }
  
  static P e2_g4(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  inline static const partial_term<P> e2_pterm_x = partial_term<P>(
      coefficient_type::div, e2_g1, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e2_pterm_v1 = partial_term<P>(
      coefficient_type::mass, e2_g2, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e2_pterm_v2 = partial_term<P>(
      coefficient_type::mass, e2_g3, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e2_pterm_v3 = partial_term<P>(
      coefficient_type::mass, e2_g4, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term_e2x =
      term<P>(false,  // time-dependent
              "e2_x", // name
              {e2_pterm_x}, imex_flag::imex_explicit);

  inline static term<P> const term_e2v1 =
      term<P>(false,  // time-dependent
              "e2_v1", // name
              {e2_pterm_v1}, imex_flag::imex_explicit);

  inline static term<P> const term_e2v2 =
      term<P>(false,  // time-dependent
              "e2_v2", // name
              {e2_pterm_v2}, imex_flag::imex_explicit);

  inline static term<P> const term_e2v3 =
      term<P>(false,  // time-dependent
              "e2_v3", // name
              {e2_pterm_v3}, imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_2 = {term_e2x, term_e2v1, term_e2v2, term_e2v3};

  // Term 3 (e3)
  // Central Part of E\cdot\grad_v f
  //

  static P E_func(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("E");
    expect(param != nullptr);
    return param->value(x, time);
  }

  static P negOne(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }
  
  static P posOne(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  inline static const partial_term<P> e3_pterm_x = partial_term<P>(
      coefficient_type::mass, E_func, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e3_pterm_v1 = partial_term<P>(
      coefficient_type::div, negOne, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static const partial_term<P> e3_pterm_v2 = partial_term<P>(
      coefficient_type::mass, posOne, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> e3_pterm_v3 = partial_term<P>(
      coefficient_type::mass, posOne, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term_e3_x =
      term<P>(true,        // time-dependent
              "term_e3_x", // name
              {e3_pterm_x}, imex_flag::imex_explicit);

  inline static term<P> const term_e3_v1 =
      term<P>(false,        // time-dependent
              "term_e3_v1", // name
              {e3_pterm_v1}, imex_flag::imex_explicit);

  inline static term<P> const term_e3_v2 =
      term<P>(false,        // time-dependent
              "term_e3_v2", // name
              {e3_pterm_v2}, imex_flag::imex_explicit);

  inline static term<P> const term_e3_v3 =
      term<P>(false,        // time-dependent
              "term_e3_v3", // name
              {e3_pterm_v3}, imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_3 = {term_e3_x, term_e3_v1, term_e3_v2, term_e3_v3};

  // Term 4 + 5
  // Penalty Part of E\cdot\grad_v f
  //

  static P MaxAbsE_func(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("MaxAbsE");
    expect(param != nullptr);
    return param->value(x, time);
  }

  inline static const partial_term<P> pterm_MaxAbsE_mass_x = partial_term<P>(
      coefficient_type::mass, MaxAbsE_func, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> pterm_div_v_downwind = partial_term<P>(
      coefficient_type::div, posOne, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);
      
  inline static term<P> const MaxAbsE_mass_x_1 =
      term<P>(true, // time-dependent
              "",   // name
              {pterm_MaxAbsE_mass_x}, imex_flag::imex_explicit);

  inline static term<P> const MaxAbsE_mass_x_2 =
      term<P>(true, // time-dependent
              "",   // name
              {pterm_MaxAbsE_mass_x}, imex_flag::imex_explicit);
              // Why exactly the same as MaxAbsE_mass_x_1?

  inline static term<P> const div_v_downwind =
      term<P>(false, // time-dependent
              "",    // name
              {pterm_div_v_downwind}, imex_flag::imex_explicit);

  // Central Part Defined in Term 3 Above (term_e3_v1; can do this due to time independence)

  inline static std::vector<term<P>> const terms_4 = {MaxAbsE_mass_x_1,
                                                      div_v_downwind,
                                                      term_e3_v2, term_e3_v3};

  inline static std::vector<term<P>> const terms_5 = {MaxAbsE_mass_x_2, term_e3_v1,
                                                      term_e3_v2, term_e3_v3};

  // Terms 3 - 5 from vlasov_lb_full_f PDE:

  // Term 3
  // v\cdot\grad_v f
  //
  static P i1_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return nu;
  }

  static P i1_g2(P const x, P const time = 0)
  {
    ignore(time);
    return x;
  }
  inline static const partial_term<P> i1_pterm_x = partial_term<P>(
      coefficient_type::mass, i1_g1, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> i1_pterm_v = partial_term<P>(
      coefficient_type::div, i1_g2, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term_i1x =
      term<P>(false,  // time-dependent
              "I1_x", // name
              {i1_pterm_x}, imex_flag::imex_implicit);

  inline static term<P> const term_i1v =
      term<P>(false,  // time-dependent
              "I1_v", // name
              {i1_pterm_v}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_6 = {term_i1x, term_i1v};

  // Term 4
  // -u\cdot\grad_v f
  //
  static P i2_g1(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("u");
    expect(param != nullptr);
    return -param->value(x, time);
  }

  static P i2_g2(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return nu;
  }

  inline static const partial_term<P> i2_pterm_x = partial_term<P>(
      coefficient_type::mass, i2_g1, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> i2_pterm_v = partial_term<P>(
      coefficient_type::div, i2_g2, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term_i2x =
      term<P>(false,  // time-dependent
              "I2_x", // name
              {i2_pterm_x}, imex_flag::imex_implicit);

  inline static term<P> const term_i2v =
      term<P>(false,  // time-dependent
              "I2_v", // name
              {i2_pterm_v}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_7 = {term_i2x, term_i2v};

  // Term 5
  // div_v(th\grad_v f)
  //
  // Split by LDG
  //
  // div_v(th q)
  // q = \grad_v f
  static P i3_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  static P i3_g2(P const x, P const time = 0)
  {
    auto param = param_manager.get_parameter("theta");
    expect(param != nullptr);
    return param->value(x, time) * nu;
  }

  inline static const partial_term<P> i3_pterm_x1 = partial_term<P>(
      coefficient_type::mass, i3_g1, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static const partial_term<P> i3_pterm_x2 = partial_term<P>(
      coefficient_type::mass, i3_g2, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term_i3x =
      term<P>(false,  // time-dependent
              "I3_x", // name
              {i3_pterm_x1, i3_pterm_x2}, imex_flag::imex_implicit);

  static P i3_g3(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  static P i3_g4(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  inline static const partial_term<P> i3_pterm_v1 = partial_term<P>(
      coefficient_type::div, i3_g3, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static const partial_term<P> i3_pterm_v2 = partial_term<P>(
      coefficient_type::grad, i3_g4, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet);

  inline static term<P> const term_i3v =
      term<P>(false,  // time-dependent
              "I3_v", // name
              {i3_pterm_v1, i3_pterm_v2}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_8 = {term_i3x, term_i3v};

  inline static term_set<P> const terms_ = {terms_1, terms_2, terms_3, terms_4, terms_5};

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {};
  inline static scalar_func<P> const exact_scalar_func_               = {};

  static P get_dt_(dimension<P> const &dim)
  {
    ignore(dim);
    /* return dx; this will be scaled by CFL from command line */
    // return std::pow(0.25, dim.get_level());

    // TODO: these are constants since we want dt always based on dim 2,
    //  but there is no way to force a different dim for this function!
    // (Lmax - Lmin) / 2 ^ LevX * CFL
    return (6.0 - (-6.0)) / std::pow(2, 3);
  }

  /* problem contains no sources */
  inline static std::vector<source<P>> const sources_ = {};
};

} // namespace asgard
