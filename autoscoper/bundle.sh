#!/bin/bash

set -e
set -x

SKEL="bundle-skel"
BUILD="bundle-build"
LIB="$BUILD/Contents/MacOS/lib"

cp -r $SKEL $BUILD
cp autoscoper $BUILD/Contents/MacOS/
mkdir -p $LIB

# Copy libraries and change absolute library paths to relative paths in
# the executable
for f in `dyldinfo -dylibs autoscoper | grep '/usr/local'`
do
	cp $f $LIB/
	lib=`basename $f`
	install_name_tool -change $f "@executable_path/lib/$lib" $BUILD/Contents/MacOS/autoscoper
	lib="$LIB/$lib"
	chmod 644 $lib
	# Also copy all of the library's dependencies
	for f in `dyldinfo -dylibs $lib | grep '/usr/local'`
	do
		cp $f $LIB/
		chmod 644 $LIB/`basename $f`
	done
done

for f in /usr/local/opt/icu4c/lib/*.1.dylib
do
	cp $f $LIB/
	chmod 644 $LIB/`basename $f`
done

#for f in $LIB/*.dylib
#do
#	lib=$f
#	# Also copy all of the library's dependencies
#	for f in `dyldinfo -dylibs $lib | grep '/usr/local'`
#	do
#		cp $f $LIB/
#		chmod 644 $LIB/`basename $f`
#	done
#done
#
# Change absolute library paths to relative paths in each library
for f in $LIB/*.dylib
do
	for lib in `dyldinfo -dylibs $f | grep '/usr/local'`
	do
		install_name_tool -change $lib "@loader_path/`basename $lib`" $f
	done
done

# Fix missing symlinks
SAVE_PWD=$PWD
cd $LIB
rm libgdkglext-x11-1.0.0.dylib 
ln -s libgdkglext-x11-1.0.0.0.0.dylib libgdkglext-x11-1.0.0.dylib
for f in /usr/local/opt/icu4c/lib/*.1.dylib
do
	lib=`basename $f`
	ln -s $lib ${lib%1.dylib}dylib
done
cd $SAVE_PWD

# Zip
rm -rf "XROMM Autoscoper.app" "XROMM Autoscoper.zip"
mv $BUILD "XROMM Autoscoper.app"
zip -r "XROMM Autoscoper.zip" "XROMM Autoscoper.app"

