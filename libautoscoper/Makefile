TARGET = libautoscoper.a

CXX = g++
CXXFLAGS = -Wall -g -O2 -fPIC
AR = ar
ARFLAGS = cru

CXXFILES = $(shell ls *.cpp)
CLFILES = $(shell ls *.cl)

CLDIR = cl
KERNELS = $(CLFILES:%.cl=$(CLDIR)/%.cl.h)

OBJDIR = obj
OBJS = $(CXXFILES:%.cpp=$(OBJDIR)/%.cpp.o)

INCLUDES = -I. -I$(CLDIR)

.PHONY:	clean

all:	$(TARGET)

$(TARGET) : $(KERNELS) $(OBJS)
	$(AR) $(ARFLAGS) $(TARGET) $(OBJS)

$(OBJDIR)/%.cpp.o: %.cpp $(KERNELS)
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(CLDIR)/%.cl.h: %.cl
	mkdir -p $(CLDIR)
	./cl-compile $< $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o $(CLDIR)/*.h

