TARGET = libautoscoper.a

CC = gcc
CXX = g++
CFLAGS = -Wall -g -O2 -fPIC -m64
CXXFLAGS = $(CFLAGS)
NVCC = nvcc
NVCCFLAGS = -m64
AR = ar
ARFLAGS = cr

CXXFILES = $(shell ls *.cpp)
CUFILES = $(shell ls *.cu)
CLFILES = $(shell ls *.cl)

OBJDIR = obj
OBJS = \
	$(CXXFILES:%.cpp=$(OBJDIR)/%.cpp.o) \
	$(CUFILES:%.cu=$(OBJDIR)/%.cu.o) \
	$(CLFILES:%.cl=$(OBJDIR)/%.cl.o)

INCLUDES = -I. -Icutil -I$(CUDA_DIR)/include

.PHONY:	check-cuda clean

all:	$(TARGET)

check-cuda:
ifndef CUDA_DIR
	$(error CUDA_DIR is undefined)
endif
	@ echo "CUDA_DIR = $(CUDA_DIR)"

$(TARGET) : check-cuda $(OBJS)
	@ echo $(OBJS)
	$(AR) $(ARFLAGS) $(TARGET) $(OBJS)

$(OBJDIR)/%.cpp.o: %.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.cu.o: %.cu
	mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.cl.o: %.cl
	mkdir -p $(OBJDIR)
	./cl-compile $< $(OBJDIR)/$<.c
	$(CC) $(CFLAGS) -c $(OBJDIR)/$<.c -o $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*

