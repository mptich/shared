set -e

VERSION=$1
PARALLEL=$2

git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout $VERSION
cd ..
git clone https://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout $VERSION
cd ..

cd ~/opencv
mkdir build
cd build
cmake $PARALLEL -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON \
    -D WITH_CUDA=OFF \
    -D WITH_TBB=ON \
    ..

sudo make install
sudo ldconfig
