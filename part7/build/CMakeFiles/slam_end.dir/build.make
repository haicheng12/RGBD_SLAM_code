# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/RGBD_SLAM_code/part7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/RGBD_SLAM_code/part7/build

# Include any dependencies generated for this target.
include CMakeFiles/slam_end.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/slam_end.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/slam_end.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/slam_end.dir/flags.make

CMakeFiles/slam_end.dir/src/slam_base.cpp.o: CMakeFiles/slam_end.dir/flags.make
CMakeFiles/slam_end.dir/src/slam_base.cpp.o: ../src/slam_base.cpp
CMakeFiles/slam_end.dir/src/slam_base.cpp.o: CMakeFiles/slam_end.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/RGBD_SLAM_code/part7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/slam_end.dir/src/slam_base.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/slam_end.dir/src/slam_base.cpp.o -MF CMakeFiles/slam_end.dir/src/slam_base.cpp.o.d -o CMakeFiles/slam_end.dir/src/slam_base.cpp.o -c /home/ubuntu/RGBD_SLAM_code/part7/src/slam_base.cpp

CMakeFiles/slam_end.dir/src/slam_base.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slam_end.dir/src/slam_base.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/RGBD_SLAM_code/part7/src/slam_base.cpp > CMakeFiles/slam_end.dir/src/slam_base.cpp.i

CMakeFiles/slam_end.dir/src/slam_base.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slam_end.dir/src/slam_base.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/RGBD_SLAM_code/part7/src/slam_base.cpp -o CMakeFiles/slam_end.dir/src/slam_base.cpp.s

CMakeFiles/slam_end.dir/src/slam_end.cpp.o: CMakeFiles/slam_end.dir/flags.make
CMakeFiles/slam_end.dir/src/slam_end.cpp.o: ../src/slam_end.cpp
CMakeFiles/slam_end.dir/src/slam_end.cpp.o: CMakeFiles/slam_end.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/RGBD_SLAM_code/part7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/slam_end.dir/src/slam_end.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/slam_end.dir/src/slam_end.cpp.o -MF CMakeFiles/slam_end.dir/src/slam_end.cpp.o.d -o CMakeFiles/slam_end.dir/src/slam_end.cpp.o -c /home/ubuntu/RGBD_SLAM_code/part7/src/slam_end.cpp

CMakeFiles/slam_end.dir/src/slam_end.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slam_end.dir/src/slam_end.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/RGBD_SLAM_code/part7/src/slam_end.cpp > CMakeFiles/slam_end.dir/src/slam_end.cpp.i

CMakeFiles/slam_end.dir/src/slam_end.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slam_end.dir/src/slam_end.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/RGBD_SLAM_code/part7/src/slam_end.cpp -o CMakeFiles/slam_end.dir/src/slam_end.cpp.s

# Object files for target slam_end
slam_end_OBJECTS = \
"CMakeFiles/slam_end.dir/src/slam_base.cpp.o" \
"CMakeFiles/slam_end.dir/src/slam_end.cpp.o"

# External object files for target slam_end
slam_end_EXTERNAL_OBJECTS =

slam_end: CMakeFiles/slam_end.dir/src/slam_base.cpp.o
slam_end: CMakeFiles/slam_end.dir/src/slam_end.cpp.o
slam_end: CMakeFiles/slam_end.dir/build.make
slam_end: /usr/lib/x86_64-linux-gnu/libboost_system.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_thread.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_regex.so
slam_end: /usr/local/lib/libpcl_common.so
slam_end: /usr/local/lib/libpcl_octree.so
slam_end: /usr/lib/libOpenNI.so
slam_end: /usr/lib/libOpenNI2.so
slam_end: /usr/local/lib/libpcl_io.so
slam_end: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
slam_end: /usr/local/lib/libpcl_kdtree.so
slam_end: /usr/local/lib/libpcl_search.so
slam_end: /usr/local/lib/libpcl_sample_consensus.so
slam_end: /usr/local/lib/libpcl_filters.so
slam_end: /usr/local/lib/libpcl_features.so
slam_end: /usr/local/lib/libpcl_ml.so
slam_end: /usr/local/lib/libpcl_segmentation.so
slam_end: /usr/local/lib/libpcl_visualization.so
slam_end: /usr/lib/x86_64-linux-gnu/libqhull.so
slam_end: /usr/local/lib/libpcl_surface.so
slam_end: /usr/local/lib/libpcl_registration.so
slam_end: /usr/local/lib/libpcl_keypoints.so
slam_end: /usr/local/lib/libpcl_tracking.so
slam_end: /usr/local/lib/libpcl_recognition.so
slam_end: /usr/local/lib/libpcl_stereo.so
slam_end: /usr/local/lib/libpcl_outofcore.so
slam_end: /usr/local/lib/libpcl_people.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_system.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_thread.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
slam_end: /usr/lib/x86_64-linux-gnu/libboost_regex.so
slam_end: /usr/lib/x86_64-linux-gnu/libqhull.so
slam_end: /usr/lib/libOpenNI.so
slam_end: /usr/lib/libOpenNI2.so
slam_end: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
slam_end: /usr/local/lib/libvtkDomainsChemistryOpenGL2-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersFlowPaths-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersGeneric-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersHyperTree-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersParallelImaging-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersPoints-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersProgrammable-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersSMP-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersSelection-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersTexture-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersVerdict-7.1.so.1
slam_end: /usr/local/lib/libvtkverdict-7.1.so.1
slam_end: /usr/local/lib/libvtkGeovisCore-7.1.so.1
slam_end: /usr/local/lib/libvtkproj4-7.1.so.1
slam_end: /usr/local/lib/libvtkIOAMR-7.1.so.1
slam_end: /usr/local/lib/libvtkIOEnSight-7.1.so.1
slam_end: /usr/local/lib/libvtkIOExodus-7.1.so.1
slam_end: /usr/local/lib/libvtkIOExport-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingGL2PSOpenGL2-7.1.so.1
slam_end: /usr/local/lib/libvtkgl2ps-7.1.so.1
slam_end: /usr/local/lib/libvtkIOImport-7.1.so.1
slam_end: /usr/local/lib/libvtkIOInfovis-7.1.so.1
slam_end: /usr/local/lib/libvtklibxml2-7.1.so.1
slam_end: /usr/local/lib/libvtkIOLSDyna-7.1.so.1
slam_end: /usr/local/lib/libvtkIOMINC-7.1.so.1
slam_end: /usr/local/lib/libvtkIOMovie-7.1.so.1
slam_end: /usr/local/lib/libvtkoggtheora-7.1.so.1
slam_end: /usr/local/lib/libvtkIOPLY-7.1.so.1
slam_end: /usr/local/lib/libvtkIOParallel-7.1.so.1
slam_end: /usr/local/lib/libvtkjsoncpp-7.1.so.1
slam_end: /usr/local/lib/libvtkIOParallelXML-7.1.so.1
slam_end: /usr/local/lib/libvtkIOSQL-7.1.so.1
slam_end: /usr/local/lib/libvtksqlite-7.1.so.1
slam_end: /usr/local/lib/libvtkIOTecplotTable-7.1.so.1
slam_end: /usr/local/lib/libvtkIOVideo-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingMorphological-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingStatistics-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingStencil-7.1.so.1
slam_end: /usr/local/lib/libvtkInteractionImage-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingContextOpenGL2-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingImage-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingLOD-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingVolumeOpenGL2-7.1.so.1
slam_end: /usr/local/lib/libvtkViewsContext2D-7.1.so.1
slam_end: /usr/local/lib/libvtkViewsInfovis-7.1.so.1
slam_end: /usr/local/lib/libopencv_gapi.so.4.5.5
slam_end: /usr/local/lib/libopencv_highgui.so.4.5.5
slam_end: /usr/local/lib/libopencv_ml.so.4.5.5
slam_end: /usr/local/lib/libopencv_objdetect.so.4.5.5
slam_end: /usr/local/lib/libopencv_photo.so.4.5.5
slam_end: /usr/local/lib/libopencv_stitching.so.4.5.5
slam_end: /usr/local/lib/libopencv_video.so.4.5.5
slam_end: /usr/local/lib/libopencv_videoio.so.4.5.5
slam_end: /usr/local/lib/libpangolin.so
slam_end: ../Thirdparty/DBoW2/lib/libDBoW2.so
slam_end: ../Thirdparty/g2o/lib/libg2o.so
slam_end: /usr/local/lib/libpcl_common.so
slam_end: /usr/local/lib/libpcl_octree.so
slam_end: /usr/local/lib/libpcl_io.so
slam_end: /usr/local/lib/libpcl_kdtree.so
slam_end: /usr/local/lib/libpcl_search.so
slam_end: /usr/local/lib/libpcl_sample_consensus.so
slam_end: /usr/local/lib/libpcl_filters.so
slam_end: /usr/local/lib/libpcl_features.so
slam_end: /usr/local/lib/libpcl_ml.so
slam_end: /usr/local/lib/libpcl_segmentation.so
slam_end: /usr/local/lib/libpcl_visualization.so
slam_end: /usr/local/lib/libpcl_surface.so
slam_end: /usr/local/lib/libpcl_registration.so
slam_end: /usr/local/lib/libpcl_keypoints.so
slam_end: /usr/local/lib/libpcl_tracking.so
slam_end: /usr/local/lib/libpcl_recognition.so
slam_end: /usr/local/lib/libpcl_stereo.so
slam_end: /usr/local/lib/libpcl_outofcore.so
slam_end: /usr/local/lib/libpcl_people.so
slam_end: ../Thirdparty/DBoW2/lib/libDBoW2.so
slam_end: ../Thirdparty/g2o/lib/libg2o.so
slam_end: /usr/local/lib/libvtkDomainsChemistry-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersAMR-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersParallel-7.1.so.1
slam_end: /usr/local/lib/libvtkexoIIc-7.1.so.1
slam_end: /usr/local/lib/libvtkIOGeometry-7.1.so.1
slam_end: /usr/local/lib/libvtkIONetCDF-7.1.so.1
slam_end: /usr/local/lib/libvtkNetCDF_cxx-7.1.so.1
slam_end: /usr/local/lib/libvtkNetCDF-7.1.so.1
slam_end: /usr/local/lib/libvtkhdf5_hl-7.1.so.1
slam_end: /usr/local/lib/libvtkhdf5-7.1.so.1
slam_end: /usr/local/lib/libvtkParallelCore-7.1.so.1
slam_end: /usr/local/lib/libvtkIOLegacy-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingOpenGL2-7.1.so.1
slam_end: /usr/lib/x86_64-linux-gnu/libXt.so
slam_end: /usr/local/lib/libvtkglew-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingMath-7.1.so.1
slam_end: /usr/local/lib/libvtkChartsCore-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingContext2D-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersImaging-7.1.so.1
slam_end: /usr/local/lib/libvtkInfovisLayout-7.1.so.1
slam_end: /usr/local/lib/libvtkInfovisCore-7.1.so.1
slam_end: /usr/local/lib/libvtkViewsCore-7.1.so.1
slam_end: /usr/local/lib/libvtkInteractionWidgets-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersHybrid-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingGeneral-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingSources-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersModeling-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingHybrid-7.1.so.1
slam_end: /usr/local/lib/libvtkIOImage-7.1.so.1
slam_end: /usr/local/lib/libvtkDICOMParser-7.1.so.1
slam_end: /usr/local/lib/libvtkmetaio-7.1.so.1
slam_end: /usr/local/lib/libvtkpng-7.1.so.1
slam_end: /usr/local/lib/libvtktiff-7.1.so.1
slam_end: /usr/local/lib/libvtkjpeg-7.1.so.1
slam_end: /usr/lib/x86_64-linux-gnu/libm.so
slam_end: /usr/local/lib/libvtkInteractionStyle-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersExtraction-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersStatistics-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingFourier-7.1.so.1
slam_end: /usr/local/lib/libvtkalglib-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingAnnotation-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingColor-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingVolume-7.1.so.1
slam_end: /usr/local/lib/libvtkImagingCore-7.1.so.1
slam_end: /usr/local/lib/libvtkIOXML-7.1.so.1
slam_end: /usr/local/lib/libvtkIOXMLParser-7.1.so.1
slam_end: /usr/local/lib/libvtkIOCore-7.1.so.1
slam_end: /usr/local/lib/libvtkexpat-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingLabel-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingFreeType-7.1.so.1
slam_end: /usr/local/lib/libvtkRenderingCore-7.1.so.1
slam_end: /usr/local/lib/libvtkCommonColor-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersGeometry-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersSources-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersGeneral-7.1.so.1
slam_end: /usr/local/lib/libvtkCommonComputationalGeometry-7.1.so.1
slam_end: /usr/local/lib/libvtkFiltersCore-7.1.so.1
slam_end: /usr/local/lib/libvtkCommonExecutionModel-7.1.so.1
slam_end: /usr/local/lib/libvtkCommonDataModel-7.1.so.1
slam_end: /usr/local/lib/libvtkCommonTransforms-7.1.so.1
slam_end: /usr/local/lib/libvtkCommonMisc-7.1.so.1
slam_end: /usr/local/lib/libvtkCommonMath-7.1.so.1
slam_end: /usr/local/lib/libvtkCommonSystem-7.1.so.1
slam_end: /usr/local/lib/libvtkCommonCore-7.1.so.1
slam_end: /usr/local/lib/libvtksys-7.1.so.1
slam_end: /usr/local/lib/libvtkfreetype-7.1.so.1
slam_end: /usr/local/lib/libvtkzlib-7.1.so.1
slam_end: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
slam_end: /usr/local/lib/libopencv_dnn.so.4.5.5
slam_end: /usr/local/lib/libopencv_calib3d.so.4.5.5
slam_end: /usr/local/lib/libopencv_features2d.so.4.5.5
slam_end: /usr/local/lib/libopencv_flann.so.4.5.5
slam_end: /usr/local/lib/libopencv_imgproc.so.4.5.5
slam_end: /usr/local/lib/libopencv_core.so.4.5.5
slam_end: /usr/lib/x86_64-linux-gnu/libGL.so
slam_end: /usr/lib/x86_64-linux-gnu/libGLU.so
slam_end: /usr/lib/x86_64-linux-gnu/libGLEW.so
slam_end: /usr/lib/x86_64-linux-gnu/libSM.so
slam_end: /usr/lib/x86_64-linux-gnu/libICE.so
slam_end: /usr/lib/x86_64-linux-gnu/libX11.so
slam_end: /usr/lib/x86_64-linux-gnu/libXext.so
slam_end: /usr/lib/x86_64-linux-gnu/libpython3.6m.so
slam_end: /usr/lib/x86_64-linux-gnu/libdc1394.so
slam_end: /usr/lib/x86_64-linux-gnu/libavcodec.so
slam_end: /usr/lib/x86_64-linux-gnu/libavformat.so
slam_end: /usr/lib/x86_64-linux-gnu/libavutil.so
slam_end: /usr/lib/x86_64-linux-gnu/libswscale.so
slam_end: /usr/lib/libOpenNI.so
slam_end: /usr/lib/libOpenNI2.so
slam_end: /usr/lib/x86_64-linux-gnu/libpng.so
slam_end: /usr/lib/x86_64-linux-gnu/libz.so
slam_end: /usr/lib/x86_64-linux-gnu/libjpeg.so
slam_end: /usr/lib/x86_64-linux-gnu/libtiff.so
slam_end: /usr/lib/x86_64-linux-gnu/libIlmImf.so
slam_end: CMakeFiles/slam_end.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/RGBD_SLAM_code/part7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable slam_end"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/slam_end.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/slam_end.dir/build: slam_end
.PHONY : CMakeFiles/slam_end.dir/build

CMakeFiles/slam_end.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/slam_end.dir/cmake_clean.cmake
.PHONY : CMakeFiles/slam_end.dir/clean

CMakeFiles/slam_end.dir/depend:
	cd /home/ubuntu/RGBD_SLAM_code/part7/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/RGBD_SLAM_code/part7 /home/ubuntu/RGBD_SLAM_code/part7 /home/ubuntu/RGBD_SLAM_code/part7/build /home/ubuntu/RGBD_SLAM_code/part7/build /home/ubuntu/RGBD_SLAM_code/part7/build/CMakeFiles/slam_end.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/slam_end.dir/depend
