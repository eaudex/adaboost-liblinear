CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC
LIBS = blas/blas.a
SHVER = 1
OS = $(shell uname)
#LIBS = -lblas

all: train predict train-adaboost-linear predict-adaboost-linear train-adaboost-stump predict-adaboost-stump

lib: linear.o tron.o blas/blas.a
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,liblinear.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,liblinear.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} linear.o tron.o blas/blas.a -o liblinear.so.$(SHVER)

train: tron.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train train.c tron.o linear.o $(LIBS)

train-adaboost-linear: tron.o linear.o train-adaboost-linear.c blas/blas.a
	$(CXX) $(CFLAGS) -o train-adaboost-linear train-adaboost-linear.c tron.o linear.o $(LIBS)
predict-adaboost-linear: tron.o linear.o predict-adaboost-linear.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict-adaboost-linear predict-adaboost-linear.c tron.o linear.o $(LIBS)

train-adaboost-stump: train-adaboost-stump.c
	$(CXX) $(CFLAGS) -o train-adaboost-stump train-adaboost-stump.c
predict-adaboost-stump: predict-adaboost-stump.c
	$(CXX) $(CFLAGS) -o predict-adaboost-stump predict-adaboost-stump.c

predict: tron.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c tron.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ tron.o linear.o train predict liblinear.so.$(SHVER)
	rm -f train-adaboost-linear predict-adaboost-linear train-adaboost-stump predict-adaboost-stump
