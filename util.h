#ifndef _UTIL_H
#define _UTIL_H

#include <stdlib.h>

//template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#define Malloc(type,n) (type *)malloc(((size_t)(n))*sizeof(type))

void swap(int* x, int* y) {
	int t = *x;
	*x = *y;
	*y = t;
}
/*
void swap(double* x, double* y) {
	double t = *x;
	*x = *y;
	*y = t;
}
*/

#endif /* _UTIL_H */
