all:
	cd libautoscoper; make -j 2;
	cd autoscoper; make -j 2;

clean:
	cd libautoscoper; make clean;
	cd autoscoper; make clean;
