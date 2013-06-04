CXX = g++
CXXFLAGS = -Wall -g -O2 -fPIC -DDEBUG
AR = ar
ARFLAGS = cru

CXXFILES = $(shell ls *.cpp)
CLFILES = $(shell ls *.cl)

CLDIR = cl
KERNELS = $(CLFILES:%.cl=$(CLDIR)/%.cl.h)

OBJDIR = obj
OBJS = $(CXXFILES:%.cpp=$(OBJDIR)/%.cpp.o)

INCLUDES = -I. -I$(CLDIR) -I/opt/X11/include

.PHONY:	clean

all:	libautoscoper.a

libautoscoper.a: $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(CLDIR)/%.cl.h: %.cl
	mkdir -p $(CLDIR)
	./cl-compile $< $@

$(OBJDIR)/%.cpp.o: %.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/Compositor.cpp.o: Compositor.cpp $(CLDIR)/Compositor.cl.h 
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/ContrastFilter.cpp.o: ContrastFilter.cpp $(CLDIR)/ContrastFilter.cl.h 
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/GaussianFilter.cpp.o: GaussianFilter.cpp $(CLDIR)/GaussianFilter.cl.h 
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/Ncc.cpp.o: Ncc.cpp $(CLDIR)/Ncc.cl.h $(CLDIR)/NccSum.cl.h
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/RadRenderer.cpp.o: RadRenderer.cpp $(CLDIR)/RadRenderer.cl.h 
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/RayCaster.cpp.o: RayCaster.cpp $(CLDIR)/RayCaster.cl.h 
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/SharpenFilter.cpp.o: SharpenFilter.cpp $(CLDIR)/SharpenFilter.cl.h 
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/SobelFilter.cpp.o: SobelFilter.cpp $(CLDIR)/SobelFilter.cl.h 
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o $(CLDIR)/*.h

