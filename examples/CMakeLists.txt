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

add_executable(anidiff anidiff.cc)
target_link_libraries(anidiff
  mschol
  ${GMP_LIBRARIES}
  ${OpenCV_LIBRARIES}
  )


add_executable(anidiff2 anidiff2.cc)
target_link_libraries(anidiff2
  mschol
  ${GMP_LIBRARIES}
  )

add_executable(mat2cv mat2cv.cc)
target_link_libraries(mat2cv
    ${OpenCV_LIBRARIES}
    )
  
