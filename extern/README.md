If not provided by the system, download and install the Eigen library:
```
git clone https://gitlab.com/libeigen/eigen.git eigen
cd eigen
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=../../eigen_lib
make install
```
