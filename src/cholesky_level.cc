#include "cholesky_level.h"

#include "macro.h"

using namespace std;
using namespace Eigen;

namespace mschol {

VectorXd chol_level::calc_supp_scale(const double *Vol) const {
  const size_t node_num = nods_.cols();
  VectorXd l = VectorXd::Zero(node_num);
  
  if ( mesh_type_ == "trig" ) {
    ASSERT(nods_.rows() == 2 && Vol);
    const double h = sqrt(*Vol/node_num);
    cout << "--- length scale: " << sqrt(2.0)*h << endl;
    l = sqrt(2.0)*h*VectorXd::Ones(node_num);
  } else if ( mesh_type_ == "tets" ) {
    ASSERT(nods_.rows() == 3 && Vol);
    const double h = cbrt(*Vol/node_num);
    cout << "--- length scale: " << sqrt(3.0)*h << endl;
    l = sqrt(3.0)*h*VectorXd::Ones(node_num);
  } else {
    bool mesh_type_is_not_supported = false;
    ASSERT(mesh_type_is_not_supported);
  }

  return l;
}

}
