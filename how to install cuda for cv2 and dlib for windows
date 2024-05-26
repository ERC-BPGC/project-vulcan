Cuda version: 11.2 download from nvidia website.
download zlib and add .dll to path
CUDNN v 8.1.1.33 download from nvidia website. follow installation instructions from website.
dont forget to add cudnn to path.
check by typing nvcc --version in cmd.

dlib download 19.23 from github
opencv 4.x from github.
opencv_contrib same v from github.

download cmake as a software, not with pip
download visual studio 2019.
install visual studio build tools 2017 and 2019. select c++ tools. make sure tools for cmake is ticked.


now we build dlib.
use cmake to build dlib.

use source code as dlib folder path
where to build as dlib folder path / build. create a build folder by yourself.

Make sure you dont use jionet now onwards.
If you see blank screen, press configure. use visual studio 2017 and set x64 in next blank.
now change DLIB_USE_CUDA to ON or ticked and same for DLIB_LINK_WITH_SQLITE3.

now press configure and if you see "DLIB WILL USE CUDA", congrats, you are almost done (hopefully). Delete older installed versions of dlib with pip or go into site_packages folder and delete it.
Now press generate and close cmake once this is done.
Go to build folder that you create. open cmd there and type "cmake ." . after this do cd.. or go to dlib folder and open cmd. type "python setup.py install"

you can check by closing all python codes and opening a python terminal and writing in shell:

import dlib
dlib.DLIB_USE_CUDA
if you get true, congratulations.


moving to open cv, first uninstall all.
https://www.youtube.com/watch?v=d8Jx6zO1yw0
follow this video step by step.

hope it works.

I have used python 3.7.4, windows 11, nvidia gpu rtx 3050, intel i7 (x64).
