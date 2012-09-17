TARGETS = filters

CXX = g++
CXXFLAGS = -Wall -g -O2 -fPIC -m64
INCLUDES = -I../ -I../cutil -I$(CUDA_DIR)/include
LIBS = ../libautoscoper.a -ltiff -L$(CUDA_DIR)/lib -lcudart

.PHONY:	check-cuda clean

all:	$(TARGETS)

check-cuda:
ifndef CUDA_DIR
	$(error CUDA_DIR is undefined)
endif
	@ echo "CUDA_DIR = $(CUDA_DIR)"

filters: check-cuda filters.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) filters.cpp -o $@ $(LIBS)

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o
