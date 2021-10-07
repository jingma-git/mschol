file(GLOB source *.cc *.c)

add_library(mschol SHARED ${source})
target_link_libraries(mschol
  ${BLAS_LAPACK_LIBS}
  )
