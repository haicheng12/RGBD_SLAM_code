add_executable(circle_fit circle_fit.cpp)
set_target_properties(circle_fit PROPERTIES OUTPUT_NAME circle_fit)
target_link_libraries(circle_fit core solver_dense)

add_executable(curve_fit curve_fit.cpp)
set_target_properties(curve_fit PROPERTIES OUTPUT_NAME curve_fit)
target_link_libraries(curve_fit core solver_dense)
