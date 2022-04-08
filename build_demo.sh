#!/bin/sh
g++ demo.cpp -I./include -I./ -Wl,--whole-archive,--gc-sections -L. \
	$(pwd)/liblibjpeg-turbo.a \
	$(pwd)/liblibpng.a \
	$(pwd)/liblibtiff.a \
	$(pwd)/libopencv_core.a \
	$(pwd)/libopencv_imgcodecs.a \
	$(pwd)/libopencv_imgproc.a \
	$(pwd)/libzlib.a \
	-lpthread \
	-ldl \
	-Wl,--no-whole-archive,--gc-sections \
	-o demo
