// g++ -I ../external/eigen/ eigen.cpp -o eigen

#include <iostream>
#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;

int main() {
    std::vector<double> a;
    a.push_back(1);
    a.push_back(2);

    MatrixXd m(2, a.size());
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);

    m.row(0) = Eigen::VectorXd::Map(&a[0], a.size());
    // Eigen::Map<Eigen::VectorXd> vec(&a[0], a.size());
    // m.row(0) = vec;
    std::cout << m << std::endl;
}
