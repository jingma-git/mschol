# GMP
find_package(GMP REQUIRED)

add_executable(solve_problem_2d solve_problem_2d.cc)
target_link_libraries(solve_problem_2d
  mschol
  ${GMP_LIBRARIES}
  )

add_executable(solve_problem_3d solve_problem_3d.cc)
target_link_libraries(solve_problem_3d
  mschol
  ${GMP_LIBRARIES}
  )
