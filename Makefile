all:
	cd libautoscoper; make;
	cd autoscoper; make;

clean:
	cd libautoscoper; make clean;
	cd autoscoper; make clean;
