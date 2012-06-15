all:
	cd libautoscoper; make -j 4;
	cd autoscoper; make -j 4;

clean:
	cd libautoscoper; make clean;
	cd autoscoper; make clean;
