include_directories(/usr/include/ni )
include_directories(/usr/lib)
include_directories(/usr/include/boost)
set(OPENNI_LINKS -I/usr/include/ni -lnimCodecs -lnimRecorder -lOpenNI -lnimMockNodes -lOpenNI.jni -lGL -lglut )
set(HDF5_LINKS -I/usr/lib -lhdf5 -lhdf5_cpp)
set(BOOST_LINKS -I/usr/lib -lboost_system -lboost_filesystem )
cmake_minimum_required( VERSION 2.8 )

project( ARC1.2  )
find_package( OpenCV REQUIRED )
add_executable( ARC1.2 ARC1.2.cpp )
target_link_libraries( ARC1.2 ${OpenCV_LIBS} ${OPENNI_LINKS} ${HDF5_LINKS} 
  ${BOOST_LINKS} )
