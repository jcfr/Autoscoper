all:
	cd libautoscoper; make;
	cd autoscoper; make;
	mkdir bin; cp autoscoper/autoscoper bin/;

clean:
	cd libautoscoper; make clean;
	cd autoscoper; make clean;
