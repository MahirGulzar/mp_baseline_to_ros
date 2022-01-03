cmake_minimum_required(VERSION 2.8.3)
project(mp_baseline)

find_package(catkin REQUIRED)
catkin_package()

if (CATKIN_ENABLE_TESTING)
    # these dependencies are only needed for unit testing
    find_package(roslaunch REQUIRED)
    find_package(rostest REQUIRED)

endif (CATKIN_ENABLE_TESTING)

install(DIRECTORY
  config launch
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

