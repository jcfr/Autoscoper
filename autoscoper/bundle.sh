#!/bin/bash

set -e
set -x

SKEL="bundle-skel"
BUILD="bundle-build"

cp -r $SKEL $BUILD
mkdir -p $BUILD/Contents/Resources/lib
rm -f $BUILD/Contents/Resources/lib/*
cp autoscoper $BUILD/Contents/MacOS/

for f in `dyldinfo -dylibs autoscoper | grep '/usr/local'`
do
	cp $f $BUILD/Contents/Resources/lib/
	lib=`basename $f`
	install_name_tool -change $f "@executable_path/../Resources/lib/$lib" $BUILD/Contents/MacOS/autoscoper
done

#rm -rf "XROMM Autoscoper.app"
mv $BUILD "XROMM Autoscoper.app"
zip -r "XROMM Autoscoper.zip" "XROMM Autoscoper.app"

