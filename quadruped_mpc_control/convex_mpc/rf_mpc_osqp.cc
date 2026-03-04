#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/QR>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>
#include <assert.h>
// #include <mpc_osqp.cc>

#include <vector>
#define DCHECK_GT(a,b) assert((a)>(b))
#define DCHECK_EQ(a,b) assert((a)==(b))

#ifdef _WIN32
typedef __int64 qp_int64;
#else
typedef long long qp_int64;
#endif //_WIN32


//using Eigen::AngleAxisd;
using Eigen::Map;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
//using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;
#include "qpOASES.hpp"
#include "qpOASES/Types.hpp"

using qpOASES::QProblem;
  
typedef Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
    RowMajorMatrixXd;
    
constexpr int k3Dim = 3;
constexpr double kGravity = 9.8;
constexpr double kMaxScale = 10;
constexpr double kMinScale = 0.1;

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "osqp/include/ctrlc.h"
#include "osqp/include/osqp.h"

enum RFQPSolverName
{
  OSQP, QPOASES
};

MatrixXd RFAsBlockDiagonalMat(const std::vector<double>& qp_weights,
    int planning_horizon) {
    const Eigen::Map<const VectorXd> qp_weights_vec(qp_weights.data(),
        qp_weights.size());
    // Directly return the rhs will cause a TSAN failure, probably due to the
    // asDiagonal not reall copying the memory. Creates the temporary will ensure
    // copy on return.
    const MatrixXd qp_weights_mat =
        qp_weights_vec.replicate(planning_horizon, 1).asDiagonal();
    return qp_weights_mat;
}

// Converts a vector to the skew symmetric matrix form. For an input vector
// [a, b, c], the output matrix would be:
//   [ 0, -c,  b]
//   [ c,  0, -a]
//   [-b,  a,  0]
Eigen::Matrix3d HatMap(const Eigen::Vector3d& vec);

// Convert skew symmetric matrix into vector.
// Inverse of hatMap
Eigen::Vector3d VeeMap(const Eigen::Matrix3d& mat);

// Get matrices A, B, and D by linearizing dynamics
// About operating point using variation linearization
void CalculateAMat(const double dt,
                   const Matrix3d& inertia,
                   const Matrix3d& inertia_inv,
                   const Matrix3d& base_rot_mat,
                   const Vector3d& ang_vel,
                   const MatrixXd& foot_positions,
                   const MatrixXd& f_op, // (4, 3)
                   MatrixXd* a_mat_ptr,
                   MatrixXd& N, MatrixXd& N_inv);

void CalculateBMat(double dt, 
                   double inv_mass, 
                   const Matrix3d& inv_inertia,
                   const Matrix3d& base_rot_mat, 
                   const MatrixXd& foot_positions,
                   MatrixXd* b_mat_ptr);

void CalculateDMat(double dt, 
                   double inv_mass,
                   const Matrix3d& inertia,
                   const Matrix3d& inv_inertia,
                   const Vector3d& com_pos,
                   const Matrix3d& base_rot_mat,
                   const Vector3d& ang_vel,
                   const MatrixXd& foot_positions,
                   const MatrixXd& f_op, 
                   MatrixXd* d_mat_ptr,
                   MatrixXd& N, MatrixXd& N_inv);

// Calculates the dense QP formulation of the discretized space time dynamics.
void CalculateQpMats(const MatrixXd& a_mat,
                     const MatrixXd& b_mat,
                     const MatrixXd& d_mat,
                     const MatrixXd& qp_weights_single,
                     const MatrixXd& alpha_single, 
                     int horizon,
                     const Matrix3d& base_rot_mat,
                     const MatrixXd& f_op, // (4, 3)
                     const MatrixXd& f_d,  // (4, 3)
                     const VectorXd& desired_states, // (18*h, 1)
                     MatrixXd* H_mat_ptr,
                     VectorXd* g_vec_ptr);

// Constraints include linearized dynamics
// And frictin cone and force bounds constraints
void UpdateConstraintsMatrix(std::vector<double>& friction_coeff,
                             int horizon,
                             int num_legs,
                             MatrixXd& a_mat,
                             MatrixXd& b_mat,
                             MatrixXd* constraint_ptr);

void CalculateConstraintBounds(const MatrixXd& contact_state,
                               double fz_max,
                               double fz_min, 
                               double friction_coeff,
                               int horizon,
                               Vector3d& com_pos,
                               Vector3d& com_vel,
                               Vector3d& ang_vel,
                               MatrixXd& f_op,
                               MatrixXd& a_mat,
                               MatrixXd& d_mat,
                               VectorXd* constraint_lb_ptr,
                               VectorXd* constraint_ub_ptr);


// The representation-free mpc implementation as described in this paper:
//   https://ieeexplore.ieee.org/document/9321699/
// Computes the optimal deviation in feet contact forces given a desired center of mass
// trajectory and gait pattern.
class RFConvexMpc {
public:
    static constexpr int kStateDim =
        12;  // 6 dof pose + 3 dof eta + 3 dof angular velocity

    // For each foot contact force we use 4-dim cone approximation + 1 for z.
    static constexpr int kConstraintDim = 5;

    RFConvexMpc(double mass, const std::vector<double>& inertia, 
        int num_legs, int planning_horizon, double timestep,
        const std::vector<double>& qp_weights, double alpha = 1e-5, 
          RFQPSolverName qp_solver_name=QPOASES);

    virtual ~RFConvexMpc()
    {
        osqp_cleanup(workspace_);
    }

    // Angular velocity is measured in body frame
    // Everything else is in world frame unless specified otherwise
    std::vector<double> ComputeContactForces(
            std::vector<double> com_position,
            std::vector<double> com_velocity,
            std::vector<double> base_rot_mat, // (9, 1)
            std::vector<double> com_angular_velocity,
            std::vector<double> foot_force, // (12, 1)
            std::vector<double> foot_contact_states,
            std::vector<double> foot_positions_body_frame, // (12, 1) foot displacements, not "positions"
            std::vector<double> foot_friction_coeffs,
            std::vector<double> desired_com_position,
            std::vector<double> desired_com_velocity,
            std::vector<double> desired_base_rot_mat, // (9, 1)
            std::vector<double> desired_com_angular_velocity,
            std::vector<double> desired_foot_force);  // (12, `)

    // Reset the solver so that for the next optimization run the solver is
    // re-initialized.
    void ResetSolver();

private:
    const double mass_;
    const double inv_mass_;
    const Eigen::Matrix3d inertia_;
    const Eigen::Matrix3d inv_inertia_;
    const int num_legs_;
    const int planning_horizon_;
    const double timestep_;
    RFQPSolverName qp_solver_name_;
    const int action_dim_;

    // 9x3 matrix, see papers for details
    Eigen::MatrixXd N_;
    Eigen::MatrixXd N_inv_;

    // 12 x 12 diagonal matrix.
    const Eigen::MatrixXd qp_weights_single_;

    // num_legs * 3 diagonal matrix.
    const Eigen::MatrixXd alpha_single_;

    // The following matrices will be updated for every call. However, their sizes
    // can be determined at class initialization time.
    Eigen::VectorXd state_;                 // 18
    Eigen::Vector3d com_pos;
    Eigen::Vector3d com_vel;
    Eigen::Matrix3d rotation_;
    Eigen::Vector3d ang_vel;
    Eigen::MatrixXd f_op;                   // num_legs*3
    Eigen::VectorXd desired_states_;        // 18 * horizon
    Eigen::MatrixXd contact_states_;        // horizon x num_legs
    Eigen::MatrixXd foot_positions_base_;   // num_legs x 3
    Eigen::MatrixXd foot_positions_world_;  // num_legs x 3
    Eigen::VectorXd foot_friction_coeff_;   // num_legs
    
    // Dynamics: x' = Ax + Bd_u + D
    Eigen::MatrixXd a_mat_;            // 12 x 12
    Eigen::MatrixXd b_mat_;            // 12 x (num_legs * 3)
    Eigen::MatrixXd d_mat_;            // 12 x 1

    // cost = x'*H*x/2 + g'*x
    Eigen::MatrixXd H_mat_;  // Hessian of cost: (num_legs*3 + 12)*horizon x (num_legs*3 + 18)*horizon
    Eigen::VectorXd g_vec_;  // Gradient of cost: (num_legs*3 + 12)*horizon vector

    // Contains the constraint matrix and bounds.
    Eigen::MatrixXd constraint_;  // (12+5*num_legs)*horizon x (3*num_legs + 12)*horizon
    Eigen::VectorXd constraint_lb_;  // (5 * num_legs + 12) * horizon
    Eigen::VectorXd constraint_ub_;  // (5 * num_legs + 12) * horizon

    std::vector<double> qp_solution_;
    
    ::OSQPWorkspace* workspace_;
    // Whether optimizing for the first step
    bool initial_run_;
};

constexpr int RFConvexMpc::kStateDim;

Matrix3d HatMap(const Vector3d& vec) {
    Matrix3d skew_symm;
    // comma initializer is row-major by default
    skew_symm << 0, -vec(2), vec(1), vec(2), 0, -vec(0), -vec(1), vec(0), 0;
    return skew_symm;
}

Vector3d VeeMap(const Matrix3d& mat) {
    Vector3d ret_vec;
    ret_vec << mat(2, 1), mat(0, 2), mat(1, 0);
    return ret_vec;
}

void CalculateAMat(const double dt,
                   const Matrix3d& inertia,
                   const Matrix3d& inertia_inv,
                   const Matrix3d& base_rot_mat,
                   const Vector3d& ang_vel,
                   const MatrixXd& foot_positions,
                   const MatrixXd& f_op, // (4, 3)
                   MatrixXd* a_mat_ptr,
                   MatrixXd& N, MatrixXd& N_inv) {
    
    const double d = ang_vel(0);
    const double e = ang_vel(1);
    const double f = ang_vel(2);
    MatrixXd D(9, 3);
    D << 0, 0, 0, 
         e,-d, 0,
         f, 0,-d,
        -e, d, 0,
         0, 0, 0,
         0, f,-e,
        -f, 0, d,
         0,-f, e,
         0, 0, 0;

    Vector3d Mop(0, 0, 0);
    for (int i = 0; i < 4; i++) {
        // Vector3d tau_foot = HatMap(foot_positions.row(i).transpose() - com_pos) * f_op.row(i).transpose();
        Vector3d tau_foot = HatMap(foot_positions.row(i).transpose()) * f_op.row(i).transpose(); // don't have to subtract com_pos because this is already a displacement
        Mop[0] += tau_foot[0];
        Mop[1] += tau_foot[1];
        Mop[2] += tau_foot[2];
    }
    
    Eigen::RowVector3d k_trans = Mop.transpose() * base_rot_mat;
    MatrixXd F = MatrixXd::Zero(3, 9);
    F.block<1, 3>(0, 0) = k_trans;
    F.block<1, 3>(1, 3) = k_trans;
    F.block<1, 3>(2, 6) = k_trans;

    MatrixXd& a_mat = *a_mat_ptr;

    // Cx_x
    a_mat.block<3, 3>(0, 0) = Matrix3d::Identity();
    // Cx_v
    a_mat.block<3, 3>(0, 3) = Matrix3d::Identity() * dt;
    // Cv_v
    a_mat.block<3, 3>(3, 3) = Matrix3d::Identity();

    // CE_eta
    MatrixXd C_eta = Eigen::kroneckerProduct(Matrix3d::Identity(), base_rot_mat*HatMap(ang_vel)) * N + Eigen::kroneckerProduct(Matrix3d::Identity(), base_rot_mat) * D; // (9, 3)
    a_mat.block<3, 3>(6, 6) = Matrix3d::Identity() + N_inv * dt * Eigen::kroneckerProduct(Matrix3d::Identity(), base_rot_mat.transpose()) * C_eta;
    // CE_w
    MatrixXd C_w = Eigen::kroneckerProduct(Matrix3d::Identity(), base_rot_mat) * N; // (9, 3)
    a_mat.block<3, 3>(6, 9) = N_inv * dt * Eigen::kroneckerProduct(Matrix3d::Identity(), base_rot_mat.transpose()) * C_w;

    // Cw_x
    Vector3d sum_fop = f_op.colwise().sum();
    Matrix3d Cx = base_rot_mat.transpose() * HatMap(sum_fop);
    a_mat.block<3, 3>(9, 0) = dt*(inertia_inv * Cx);
    // Cw_w
    Matrix3d Cw = HatMap(inertia*ang_vel) - HatMap(ang_vel) * inertia;
    a_mat.block<3, 3>(9, 9) = dt*(inertia_inv*Cw) + Matrix3d::Identity();
    // Cw_eta
    Matrix3d Ceta = F * N - Cw * HatMap(ang_vel);
    a_mat.block<3, 3>(9, 6) = dt*(inertia_inv * Ceta);
}

void CalculateBMat(double dt, 
                   double inv_mass, 
                   const Matrix3d& inv_inertia,
                   const Matrix3d& base_rot_mat, 
                   const MatrixXd& foot_positions,
                   MatrixXd* b_mat_ptr) {

    const int num_legs = foot_positions.rows();
    MatrixXd& b_mat = *b_mat_ptr;

    for (int i = 0; i < num_legs; ++i) {
        // Cw_u
        b_mat.block<k3Dim, k3Dim>(9, i * k3Dim) =
            dt * inv_inertia * base_rot_mat.transpose() * HatMap(foot_positions.row(i));
        // Cv_u
        b_mat(3, i * k3Dim) = dt*inv_mass;
        b_mat(4, i * k3Dim + 1) = dt*inv_mass;
        b_mat(5, i * k3Dim + 2) = dt*inv_mass;
    }
}

void CalculateDMat(double dt, 
                   double inv_mass,
                   const Matrix3d& inertia,
                   const Matrix3d& inv_inertia,
                   const Vector3d& com_pos,
                   const Matrix3d& base_rot_mat,
                   const Vector3d& ang_vel,
                   const MatrixXd& foot_positions,
                   const MatrixXd& f_op, 
                   MatrixXd* d_mat_ptr,
                   MatrixXd& N, MatrixXd& N_inv) {

    Vector3d Mop(0, 0, 0);
    for (int i = 0; i < 4; i++) {
        Vector3d tau_foot = HatMap(foot_positions.row(i).transpose()) * f_op.row(i).transpose(); // don't have to subtract com_pos because this is already a displacement
        Mop[0] += tau_foot[0];
        Mop[1] += tau_foot[1];
        Mop[2] += tau_foot[2];
    }
    Vector3d sum_fop = f_op.colwise().sum();
    const int num_legs = foot_positions.rows();

    MatrixXd& d_mat = *d_mat_ptr;
    // Cv_c
    d_mat.block<3, 1>(3, 0) << dt*inv_mass * sum_fop(0), dt*inv_mass * sum_fop(1), dt*(inv_mass * sum_fop(2) - kGravity);
    // CE_c
    // matlab is col-major so we do col-major here too
    VectorXd C_c = (base_rot_mat * HatMap(ang_vel)).reshaped() - Eigen::kroneckerProduct(Matrix3d::Identity(), base_rot_mat) * N * ang_vel; // (9, 1)
    d_mat.block<3, 1>(6, 0) = dt * N_inv * Eigen::kroneckerProduct(Matrix3d::Identity(), base_rot_mat.transpose()) * C_c;
    // Cw_c
    Matrix3d Cw = HatMap(inertia*ang_vel) - HatMap(ang_vel) * inertia;
    Matrix3d Cx = base_rot_mat.transpose() * HatMap(sum_fop);
    Vector3d Cc = -HatMap(ang_vel)*inertia*ang_vel + base_rot_mat.transpose()*Mop - Cw*ang_vel - Cx*com_pos;
    d_mat.block<3, 1>(9, 0) = dt * inv_inertia * Cc;
}

void CalculateQpMats(const MatrixXd& a_mat,
                     const MatrixXd& b_mat,
                     const MatrixXd& d_mat,
                     const MatrixXd& qp_weights_single,
                     const MatrixXd& alpha_single, 
                     int horizon,
                     const Matrix3d& base_rot_mat,
                     const MatrixXd& f_op, // (4, 3)
                     const MatrixXd& f_d,  // (4, 3)
                     const VectorXd& desired_states, // (18*h, 1)
                     MatrixXd* H_mat_ptr,
                     VectorXd* g_vec_ptr) {

    const int state_dim = RFConvexMpc::kStateDim;
    const int action_dim = 12;
    const int total_dim = state_dim + action_dim;

    MatrixXd& H_mat = *H_mat_ptr;
    VectorXd& g_vec = *g_vec_ptr;

    for (int i = 0; i < horizon; i++) {
        // quadratic terms
        H_mat.block<action_dim, action_dim>(i*total_dim, i*total_dim) = alpha_single;
        H_mat.block<state_dim, state_dim>(i*total_dim + action_dim, i*total_dim + action_dim) = qp_weights_single;
        // linear terms
        g_vec.segment(i*total_dim, action_dim) = alpha_single * (f_op - f_d).reshaped<Eigen::RowMajor>(); // should be alpha_single.transpose() but alpha is constant anyway
        g_vec.segment(i*total_dim + action_dim, 6) = -qp_weights_single.block<6, 6>(0, 0) * desired_states.segment(i*18, 6); // com_pos and com_vel
        // flattened R_des stored in row-major fashion, so unrolling it out in col-major fashion gives us R_des.T    
        Matrix3d des_rot_mat_transposed = desired_states.segment(i*18 + 6, 9).reshaped(3, 3); // default is column-major
        g_vec.segment(i*total_dim + action_dim + 6, 3) = 
            qp_weights_single.block<3, 3>(6, 6) * VeeMap((des_rot_mat_transposed * base_rot_mat).log()); // rotation matrix error
        g_vec.segment(i*total_dim + action_dim + 9, 3) = -qp_weights_single.block<3, 3>(9, 9) * desired_states.segment(i*18 + 15, 3); // ang_vel
    }
}

void UpdateConstraintsMatrix(std::vector<double>& friction_coeff,
                             int horizon,
                             int num_legs,
                             MatrixXd& a_mat,
                             MatrixXd& b_mat,
                             MatrixXd* constraint_ptr) {

    const int constraint_dim = RFConvexMpc::kConstraintDim;
    const int state_dim = RFConvexMpc::kStateDim;
    const int act_dim = 12;
    const int total_dim = state_dim + act_dim;

    MatrixXd& constraint = *constraint_ptr;

    // Dynamics constraints
    for (int i = 0; i < horizon; ++i) {
        if (i == 0) {
            constraint.block<state_dim, act_dim>(0, 0) = b_mat;
            constraint.block<state_dim, state_dim>(0, act_dim) = - MatrixXd::Identity(state_dim, state_dim);
        } else {
            constraint.block<state_dim, state_dim>(i*state_dim, (i-1)*total_dim + act_dim) = a_mat;
            constraint.block<state_dim, act_dim>(i*state_dim, i*total_dim) = b_mat;
            constraint.block<state_dim, state_dim>(i*state_dim, i*total_dim + act_dim) = 
                                                                    - MatrixXd::Identity(state_dim, state_dim);
        }
        // Friction cone and force constraints
        for (int j = 0; j < num_legs; ++j) {
            int row_idx = state_dim*horizon + (i*num_legs+j) * constraint_dim;
            int col_idx = (i * total_dim) + (j * k3Dim);
            constraint.block<constraint_dim, k3Dim>(row_idx, col_idx)
                << -1, 0, friction_coeff[0],
                    1, 0, friction_coeff[1],
                    0,-1, friction_coeff[2],
                    0, 1, friction_coeff[3],
                    0, 0, 1;
        }
    }
}

void CalculateConstraintBounds(const MatrixXd& contact_state,
                               double fz_max,
                               double fz_min, 
                               double friction_coeff,
                               int horizon,
                               Vector3d& com_pos,
                               Vector3d& com_vel,
                               Vector3d& ang_vel,
                               MatrixXd& f_op,
                               MatrixXd& a_mat,
                               MatrixXd& d_mat,
                               VectorXd* constraint_lb_ptr,
                               VectorXd* constraint_ub_ptr) {

    const int constraint_dim = RFConvexMpc::kConstraintDim;
    const int state_dim = RFConvexMpc::kStateDim;
    const int act_dim = 12;
    const int total_dim = state_dim + act_dim;

    VectorXd qt(state_dim);
    qt << com_pos(0), com_pos(1), com_pos(2),
          com_vel(0), com_vel(1), com_vel(2),
          0, 0, 0,
          ang_vel(0), ang_vel(1), ang_vel(2);

    int num_legs = contact_state.cols();

    VectorXd& constraint_lb = *constraint_lb_ptr;
    VectorXd& constraint_ub = *constraint_ub_ptr;

    for (int i = 0; i < horizon; ++i) {
        if (i == 0) {
            constraint_lb.segment(0, state_dim) = -d_mat - a_mat * qt;
            constraint_ub.segment(0, state_dim) = -d_mat - a_mat * qt;
        } else{
            constraint_lb.segment(i*state_dim, state_dim) = -d_mat;
            constraint_ub.segment(i*state_dim, state_dim) = -d_mat;
        }

        // force and friction cones bounds (after dynamics = state_dim*horizon elements)
        for (int j = 0; j < num_legs; ++j) {
            const int row = state_dim*horizon + (i * num_legs + j) * constraint_dim;
            constraint_lb(row + 0) =  f_op(j, 0) - friction_coeff * f_op(j, 2);
            constraint_lb(row + 1) = -f_op(j, 0) - friction_coeff * f_op(j, 2);
            constraint_lb(row + 2) =  f_op(j, 1) - friction_coeff * f_op(j, 2);
            constraint_lb(row + 3) = -f_op(j, 1) - friction_coeff * f_op(j, 2);
            constraint_lb(row + 4) = fz_min * contact_state(i, j) - f_op(j, 2); // want delta_u + u_op = 0 when feet not in contact

            const double friction_ub =
                (friction_coeff + 1) * fz_max * contact_state(i, j);
            constraint_ub(row + 0) = friction_ub + f_op(j, 0) - friction_coeff * f_op(j, 2);
            constraint_ub(row + 1) = friction_ub - f_op(j, 0) - friction_coeff * f_op(j, 2);
            constraint_ub(row + 2) = friction_ub + f_op(j, 1) - friction_coeff * f_op(j, 2);
            constraint_ub(row + 3) = friction_ub - f_op(j, 1) - friction_coeff * f_op(j, 2);
            constraint_ub(row + 4) = fz_max * contact_state(i, j) - f_op(j, 2);
        }
    }
}

RFConvexMpc::RFConvexMpc(double mass, const std::vector<double>& inertia,
    int num_legs, int planning_horizon, double timestep,
    const std::vector<double>& qp_weights, double alpha,
      RFQPSolverName qp_solver_name)
    : mass_(mass),
    inv_mass_(1 / mass),
    inertia_(inertia.data()),
    inv_inertia_(inertia_.inverse()),
    num_legs_(num_legs),
    planning_horizon_(planning_horizon),
    timestep_(timestep),
    qp_solver_name_(qp_solver_name),
    action_dim_(num_legs* k3Dim),
    N_(9,3),
    N_inv_(3,9),
    qp_weights_single_(RFAsBlockDiagonalMat(qp_weights, 1)),
    alpha_single_(alpha*MatrixXd::Identity(action_dim_, action_dim_)),
    f_op(num_legs_, k3Dim),
    desired_states_(18 * planning_horizon),
    contact_states_(planning_horizon, num_legs),
    foot_positions_base_(num_legs, k3Dim),
    foot_positions_world_(num_legs, k3Dim),
    foot_friction_coeff_(num_legs_),
    a_mat_(kStateDim, kStateDim),
    b_mat_(kStateDim, action_dim_),
    d_mat_(kStateDim, 1),
    H_mat_((action_dim_ + kStateDim) * planning_horizon,
           (action_dim_ + kStateDim) * planning_horizon),
    g_vec_((action_dim_ + kStateDim) * planning_horizon),
    constraint_((kConstraintDim*num_legs + kStateDim) * planning_horizon,
                (action_dim_ + kStateDim) * planning_horizon),
    constraint_lb_((kConstraintDim*num_legs + kStateDim) * planning_horizon),
    constraint_ub_((kConstraintDim*num_legs + kStateDim) * planning_horizon),
    qp_solution_((action_dim_ + kStateDim) * planning_horizon),
    workspace_(0),
    initial_run_(true)

{
    assert(qp_weights.size() == (kStateDim + action_dim_));
    // We assume the input inertia is a 3x3 matrix.
    assert(inertia.size() == k3Dim * k3Dim);
    state_.setZero();
    desired_states_.setZero();
    contact_states_.setZero();
    foot_positions_base_.setZero();
    foot_positions_world_.setZero();
    foot_friction_coeff_.setZero();
    a_mat_.setZero();
    b_mat_.setZero();
    d_mat_.setZero();
    N_ << 0, 0, 0,
         0, 0, 1,
         0,-1, 0,
         0, 0,-1,
         0, 0, 0,
         1, 0, 0,
         0, 1, 0,
        -1, 0, 0,
         0, 0, 0;
    N_inv_ = N_.completeOrthogonalDecomposition().pseudoInverse();
    constraint_.setZero();
    constraint_lb_.setZero();
    constraint_ub_.setZero();
}

void RFConvexMpc::ResetSolver() { initial_run_ = true; }

std::vector<double> RFConvexMpc::ComputeContactForces(
    std::vector<double> com_position,
    std::vector<double> com_velocity,
    std::vector<double> base_rot_mat, // (9, 1)
    std::vector<double> com_angular_velocity,
    std::vector<double> foot_force, // (12, 1)
    std::vector<double> foot_contact_states,
    std::vector<double> foot_positions_body_frame, // (12, 1)
    std::vector<double> foot_friction_coeffs,
    std::vector<double> desired_com_position,
    std::vector<double> desired_com_velocity,
    std::vector<double> desired_base_rot_mat, // (9, 1)
    std::vector<double> desired_com_angular_velocity,
    std::vector<double> desired_foot_force) { // (12, 1)

    std::vector<double> error_result;
    
    // operating point variables
    DCHECK_EQ(base_rot_mat.size(), 9);
    DCHECK_EQ(foot_force.size(), k3Dim * num_legs_);
    // Eigen::Map is column-major so we transpose
    f_op = Eigen::Map<const MatrixXd>(foot_force.data(), k3Dim, num_legs_).transpose();
    com_pos = Eigen::Map<const Vector3d>(com_position.data());
    com_vel = Eigen::Map<const Vector3d>(com_velocity.data());
    rotation_ = Eigen::Map<const Matrix3d>(base_rot_mat.data()).transpose(); // base_rot_mat saved in row-major order but Eigen stores in col-major
    ang_vel = Eigen::Map<const Vector3d>(com_angular_velocity.data());

    // Compute feet displacements from CoM in world frame
    DCHECK_EQ(foot_positions_body_frame.size(), k3Dim * num_legs_);
    foot_positions_base_ = Eigen::Map<const MatrixXd>(foot_positions_body_frame.data(), k3Dim, num_legs_).transpose();
    for (int i = 0; i < num_legs_; ++i) {
        foot_positions_world_.row(i) = rotation_ * foot_positions_base_.row(i).transpose();
    }

    // Prepare the current and desired state vectors of length kStateDim *
    // planning_horizon.
    DCHECK_EQ(com_velocity.size(), k3Dim);
    DCHECK_EQ(com_angular_velocity.size(), k3Dim);

    for (int i = 0; i < planning_horizon_; ++i) {
        // Position
        desired_states_[i * 18 + 0] =
            timestep_ * (i + 1) * desired_com_velocity[0];
        desired_states_[i * 18 + 1] =
            timestep_ * (i + 1) * desired_com_velocity[1];
        desired_states_[i * 18 + 2] = desired_com_position[2];

        // Velocity
        desired_states_[i * 18 + 3] = desired_com_velocity[0];
        desired_states_[i * 18 + 4] = desired_com_velocity[1];
        desired_states_[i * 18 + 5] = 0; // Prefer to stablize the body height.

        // Rotation matrix
        desired_states_.segment(i*18 + 6, 9) = Eigen::Map<VectorXd>(desired_base_rot_mat.data(), 9); // row-major

        // Prefer to stablize roll and pitch.
        desired_states_[i * 18 + 15] = desired_com_angular_velocity[0];
        desired_states_[i * 18 + 16] = desired_com_angular_velocity[1];
        desired_states_[i * 18 + 17] = desired_com_angular_velocity[2];
    }

    // Get A, B, and D matrices
    CalculateAMat(timestep_, inertia_, inv_inertia_, rotation_,
                  ang_vel, foot_positions_world_, f_op, &a_mat_,
                  N_, N_inv_);

    CalculateBMat(timestep_, inv_mass_, inv_inertia_,
                  rotation_, foot_positions_world_, &b_mat_);
    
    CalculateDMat(timestep_, inv_mass_, inertia_,
                  inv_inertia_, com_pos, rotation_,
                  ang_vel, foot_positions_world_, 
                  f_op, &d_mat_, N_, N_inv_);

    // Get desired foot force
    MatrixXd f_d = Eigen::Map<const MatrixXd>(desired_foot_force.data(), k3Dim, num_legs_).transpose();
    // Vector3d f_trot_d(0., 0., kGravity*mass_/4);
    // MatrixXd f_d = f_trot_d.replicate(1, 4).transpose();
    
    // Get H and g matrices
    CalculateQpMats(a_mat_, b_mat_, d_mat_,
                    qp_weights_single_, alpha_single_, 
                    planning_horizon_, rotation_, f_op, // (4, 3)
                    f_d, desired_states_, // (18*h, 1)
                    &H_mat_, &g_vec_);
    
    // Get contact states
    contact_states_ = Eigen::Map<const MatrixXd>(foot_contact_states.data(), num_legs_, planning_horizon_).transpose();

    CalculateConstraintBounds(contact_states_, mass_ * kGravity * kMaxScale,
                              mass_ * kGravity * kMinScale, foot_friction_coeffs[0],
                              planning_horizon_, com_pos, com_vel, ang_vel, f_op,
                              a_mat_, d_mat_, &constraint_lb_, &constraint_ub_);

    
    if (qp_solver_name_ == OSQP)
    {

      UpdateConstraintsMatrix(foot_friction_coeffs, planning_horizon_, num_legs_,
                              a_mat_, b_mat_, &constraint_);

      foot_friction_coeff_ << foot_friction_coeffs[0], foot_friction_coeffs[1],
          foot_friction_coeffs[2], foot_friction_coeffs[3];

      Eigen::SparseMatrix<double, Eigen::ColMajor, qp_int64> objective_matrix = H_mat_.sparseView();
      Eigen::VectorXd objective_vector = g_vec_;
      Eigen::SparseMatrix<double, Eigen::ColMajor, qp_int64> constraint_matrix = constraint_.sparseView();

      int num_variables = constraint_.cols();
      int num_constraints = constraint_.rows();

      ::OSQPSettings settings;
      osqp_set_default_settings(&settings);
      settings.verbose = false;
      settings.warm_start = true;
      settings.polish = true;
      settings.adaptive_rho_interval = 25;
      settings.eps_abs = 1e-3;
      settings.eps_rel = 1e-3;
      
      assert(H_mat_.cols()== num_variables);
      assert(H_mat_.rows()== num_variables);
      assert(g_vec_.size()== num_variables);
      assert(constraint_lb_.size() == num_constraints);
      assert(constraint_ub_.size() == num_constraints);

      VectorXd clipped_lower_bounds = constraint_lb_.cwiseMax(-OSQP_INFTY);
      VectorXd clipped_upper_bounds = constraint_ub_.cwiseMin(OSQP_INFTY);

      ::OSQPData data;
      data.n = num_variables;
      data.m = num_constraints;

      Eigen::SparseMatrix<double, Eigen::ColMajor, qp_int64>
          objective_matrix_upper_triangle =
          objective_matrix.triangularView<Eigen::Upper>();

      ::csc osqp_objective_matrix = {
          objective_matrix_upper_triangle.outerIndexPtr()[num_variables],
          num_variables,
          num_variables,
          const_cast<qp_int64*>(objective_matrix_upper_triangle.outerIndexPtr()),
          const_cast<qp_int64*>(objective_matrix_upper_triangle.innerIndexPtr()),
          const_cast<double*>(objective_matrix_upper_triangle.valuePtr()),
          -1 };
      data.P = &osqp_objective_matrix;

      ::csc osqp_constraint_matrix = {
          constraint_matrix.outerIndexPtr()[num_variables],
          num_constraints,
          num_variables,
          const_cast<qp_int64*>(constraint_matrix.outerIndexPtr()),
          const_cast<qp_int64*>(constraint_matrix.innerIndexPtr()),
          const_cast<double*>(constraint_matrix.valuePtr()),
          -1 };
      data.A = &osqp_constraint_matrix;

      data.q = const_cast<double*>(objective_vector.data());
      data.l = clipped_lower_bounds.data();
      data.u = clipped_upper_bounds.data();

      const int return_code = 0;
      
      if (workspace_==0) {
          osqp_setup(&workspace_, &data, &settings);
          initial_run_ = false;
      }
      else {

          UpdateConstraintsMatrix(foot_friction_coeffs, planning_horizon_, num_legs_,
                                  a_mat_, b_mat_, &constraint_);

          foot_friction_coeff_ << foot_friction_coeffs[0], foot_friction_coeffs[1],
              foot_friction_coeffs[2], foot_friction_coeffs[3];
              
          c_int nnzP = objective_matrix_upper_triangle.nonZeros();

          c_int nnzA = constraint_matrix.nonZeros();

          int return_code = osqp_update_P_A(
              workspace_, objective_matrix_upper_triangle.valuePtr(), OSQP_NULL, nnzP,
              constraint_matrix.valuePtr(), OSQP_NULL, nnzA);
          
          return_code =
              osqp_update_lin_cost(workspace_, objective_vector.data());


          return_code = osqp_update_bounds(
              workspace_, clipped_lower_bounds.data(), clipped_upper_bounds.data());
      }

      if (osqp_solve(workspace_) != 0) {
          if (osqp_is_interrupted()) {
              return error_result;
          }
      }
      
      Map<VectorXd> solution(qp_solution_.data(), qp_solution_.size());
      
      if (workspace_->info->status_val== OSQP_SOLVED) {
          // solution = -Map<const VectorXd>(workspace_->solution->x, workspace_->data->n);
          solution = Map<const VectorXd>(workspace_->solution->x, workspace_->data->n);
      }
      else {
          //LOG(WARNING) << "QP does not converge";
          return error_result;
      }
      
      return qp_solution_;
    } else
    {
      
    // Solve the QP Problem using qpOASES
    UpdateConstraintsMatrix(foot_friction_coeffs, planning_horizon_, num_legs_,
                            a_mat_, b_mat_, &constraint_);

    // Put in data structure QPOASES expects
    // Hessian
    std::vector<qpOASES::real_t> hessian(H_mat_.size());
    Map<RowMajorMatrixXd>(hessian.data(), H_mat_.rows(), H_mat_.cols()) = H_mat_;
    // Gradient
    std::vector<qpOASES::real_t> g_vec(g_vec_.data(), g_vec_.data() + g_vec_.size());
    // A_constraint
    std::vector<qpOASES::real_t> a_mat(constraint_.size());
    Map<RowMajorMatrixXd>(a_mat.data(), constraint_.rows(), constraint_.cols()) = constraint_;
    // ub
    std::vector<qpOASES::real_t> a_lb(constraint_lb_.data(), constraint_lb_.data() + constraint_lb_.size());

    std::vector<qpOASES::real_t> a_ub(constraint_ub_.data(), constraint_ub_.data() + constraint_ub_.size());

    auto qp_problem = QProblem(H_mat_.rows(), constraint_.rows(), qpOASES::HST_UNKNOWN,
                               qpOASES::BT_TRUE);

    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_NONE;
    qp_problem.setOptions(options);

    int max_solver_iter = 100;

    qp_problem.init(hessian.data(), g_vec.data(), a_mat.data(), nullptr,
                    nullptr, a_lb.data(), a_ub.data(), max_solver_iter,
                    nullptr);

    std::vector<qpOASES::real_t> qp_sol(H_mat_.rows(), 0);
    qp_problem.getPrimalSolution(qp_sol.data());
    // for (auto& force : qp_sol) {
    //   force = -force;
    // }
    Map<VectorXd> qp_sol_vec(qp_sol.data(), qp_sol.size());

    for (int i = 0; i < qp_sol.size(); ++i) {
        qp_solution_[i] = qp_sol[i];
        qp_solution_[i] = qp_sol[i];
        qp_solution_[i] = qp_sol[i];
      }

    }
    return qp_solution_;
}


namespace py = pybind11;

PYBIND11_MODULE(rf_mpc_osqp, m) {
  m.doc() = R"pbdoc(
        MPC using OSQP Python Bindings
        -----------------------

        .. currentmodule:: rf_mpc_osqp

        .. autosummary::
           :toctree: _generate

    )pbdoc";

      
   py::enum_<RFQPSolverName>(m, "RFQPSolverName")
      .value("OSQP", OSQP, "OSQP")
      .value("QPOASES", QPOASES, "QPOASES")
      .export_values();
      
  py::class_<RFConvexMpc>(m, "RFConvexMpc")
      .def(py::init<double, const std::vector<double>&, int,
          int , double ,const std::vector<double>&, double,RFQPSolverName>())
      .def("compute_contact_forces", &RFConvexMpc::ComputeContactForces)
      .def("reset_solver", &RFConvexMpc::ResetSolver);

 

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
  
  m.attr("TEST") = py::int_(int(42));

  
}
