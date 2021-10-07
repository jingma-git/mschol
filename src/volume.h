#ifndef VOLUME_H
#define VOLUME_H

#include "types.h"
#include "macro.h"

namespace mschol {

class volume_calculator
{
 public:
  volume_calculator(const mati_t &cell, const matd_t &nods,
                    const std::string &mesh_type)
      : cell_(cell), nods_(nods), mesh_type_(mesh_type) {}
  double compute() const {
    if ( mesh_type_ == "tets" ) return tets();
    if ( mesh_type_ == "trig" ) return trig();
    
    bool mesh_type_is_not_supported = false;
    ASSERT(mesh_type_is_not_supported);
    return 0;
  }

 private:
  double tets() const {
    // for tetrahedral mesh in 3D
    double vol = 0;

    matd_t Dm(3, 3);
    for (size_t i = 0; i < cell_.cols(); ++i) {
      Dm.col(0) = nods_.col(cell_(1, i))-nods_.col(cell_(0, i));
      Dm.col(1) = nods_.col(cell_(2, i))-nods_.col(cell_(0, i));
      Dm.col(2) = nods_.col(cell_(3, i))-nods_.col(cell_(0, i));
      vol += std::fabs(Dm.determinant())/6.0;
    }
    
    return vol;
  }
  double trig() const {
    // for triangular mesh in 2D    
    double vol = 0;

    matd_t Dm(2, 2);
    for (size_t i = 0; i < cell_.cols(); ++i) {
      Dm.col(0) = nods_.col(cell_(1, i))-nods_.col(cell_(0, i));
      Dm.col(1) = nods_.col(cell_(2, i))-nods_.col(cell_(0, i));
      vol += std::fabs(Dm.determinant())/2.0;
    }
    
    return vol;
  }
  
 private:
  const mati_t &cell_;
  const matd_t &nods_;
  const std::string mesh_type_;
};

}

#endif
