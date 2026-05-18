// Multivariate state-space data
// 4D state, 2D measurement

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <optional>
#include <fstream>
#include <stdexcept>
#include <vector>

struct SimulationData
{
    std::vector<Eigen::Vector4d> true_states;
    std::vector<Eigen::Vector2d> measurements;
    Eigen::Matrix4d A;
    Eigen::Matrix<double, 2, 4> C;
    Eigen::Matrix4d Q;
    Eigen::Matrix2d R;
};

SimulationData generate_4d_2d_data(
    int n_steps = 100,
    double dt = 0.1,
    double process_std = 0.2,
    double measurement_std = 1.0,
    std::optional<Eigen::Vector4d> x0 = std::nullopt,
    std::optional<unsigned int> seed = std::nullopt
)
{
    Eigen::Vector4d x;

    if (!x0)
    {
        x << 0.0, 1.0, 0.0, 0.5;
    }
    else
    {
        x = *x0;
    }

    Eigen::Matrix4d A;
    A << 1.0, dt,  0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, dt,
         0.0, 0.0, 0.0, 1.0;

    Eigen::Matrix<double, 2, 4> C;
    C << 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0;

    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt2 * dt2;

    Eigen::Matrix4d Q;
    Q << dt4 / 4.0, dt3 / 2.0, 0.0,       0.0,
         dt3 / 2.0, dt2,       0.0,       0.0,
         0.0,       0.0,       dt4 / 4.0, dt3 / 2.0,
         0.0,       0.0,       dt3 / 2.0, dt2;

    Q *= process_std * process_std;

    Eigen::Matrix2d R =
        measurement_std * measurement_std * Eigen::Matrix2d::Identity();

    std::vector<Eigen::Vector4d> true_states;
    std::vector<Eigen::Vector2d> measurements;

    std::mt19937 gen;
    if (seed)
        gen.seed(*seed);
    else
        gen.seed(std::random_device{}());

    std::normal_distribution<double> normal(0.0, 1.0);

    Eigen::Matrix4d L_Q = Q.llt().matrixL();
    Eigen::Matrix2d L_R = R.llt().matrixL();

    for (int i = 0; i < n_steps; ++i)
    {
        Eigen::Vector4d z_w;
        z_w << normal(gen), normal(gen), normal(gen), normal(gen);

        Eigen::Vector2d z_v;
        z_v << normal(gen), normal(gen);

        Eigen::Vector4d w = L_Q * z_w;
        Eigen::Vector2d v = L_R * z_v;

        x = A * x + w;
        Eigen::Vector2d y = C * x + v;

        true_states.push_back(x);
        measurements.push_back(y);
    }

    return SimulationData{
        true_states,
        measurements,
        A,
        C,
        Q,
        R
    };
}

std::vector<Eigen::Vector4d> kalman_filter (
    const std::vector<Eigen::Vector2d> &measurments, 
    const Eigen::Matrix4d &A, 
    const Eigen::Matrix<double, 2, 4> &C, 
    const Eigen::Matrix4d &Q, 
    const Eigen::Matrix2d &R
)
{
    // Initial Estimates
    Eigen::Vector4d x_hat;
    x_hat << measurments[0](0), 0.0, measurments[0](1), 0.0;
    Eigen::Matrix4d P = Eigen::Matrix4d::Identity();

    std::vector<Eigen::Vector4d> estimates;

    Eigen::Matrix4d I = Eigen::Matrix4d::Identity();

    //Iterate over all measurments
    // Alternate (GPT): for (const Eigen::Vector2d& y : measurements)
    Eigen::Matrix<double, 4, 2> K;
    Eigen::Matrix2d S;
    for(int i = 0 ; i < measurments.size() ; i++)
    {
        Eigen::Vector2d y = measurments[i];

        //Step 1 - Prediction
        //Extrapolate the state
        x_hat = A * x_hat;

        // Covariance Matrix (Uncertainty) Extrapolation
        P = A * P * A.transpose();

        // Step 2 - Update
        // Compute Kalman Gain
        // Define innovation covariance (S)
        S = C * P * C.transpose() + R;
        K = P * C.transpose() * S.inverse();

        // Update the prediction with measurment
        x_hat = x_hat + K * (y - C * x_hat);

        // Update the estimated uncertainty
        P = (I - K * C) * P * (I - K * C).transpose() + K * R * K.transpose();

        // array not mutably unlike in numpy. No copy needed.
        estimates.push_back(x_hat);
    }

    return estimates;
}


void save_kalman_csv(
    const std::string& filename,
    const std::vector<Eigen::Vector4d>& true_states,
    const std::vector<Eigen::Vector2d>& measurements,
    const std::vector<Eigen::Vector4d>& estimates
)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    file << "px_true,vx_true,py_true,vy_true,"
         << "px_meas,py_meas,"
         << "px_est,vx_est,py_est,vy_est\n";

    for (size_t i = 0; i < true_states.size(); ++i)
    {
        file << true_states[i](0) << ","
             << true_states[i](1) << ","
             << true_states[i](2) << ","
             << true_states[i](3) << ","
             << measurements[i](0) << ","
             << measurements[i](1) << ","
             << estimates[i](0) << ","
             << estimates[i](1) << ","
             << estimates[i](2) << ","
             << estimates[i](3) << "\n";
    }
}

int main() {
    auto[true_states, measurements, A, C, Q, R] = generate_4d_2d_data();

    std::vector<Eigen::Vector4d> estimates;
    estimates = kalman_filter(measurements, A, C, Q, R);

    save_kalman_csv(
        "kalman_output.csv",
        true_states,
        measurements,
        estimates
    );

    return 0;
}
