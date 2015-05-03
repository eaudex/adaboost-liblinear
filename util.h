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

void permute(int* perm, int len) {
	int i;
	for (i=0; i<len; ++i) {
		int j = i+rand()%(len-i);
		swap(&perm[i], &perm[j]);
	}
}

void sample_without_replacement(int* perm, int len, int size) {
	int i;
	for (i=0; i<size; ++i) {
		int j = i+rand()%(len-i);
		swap(&perm[i], &perm[j]);
	}
}

void sample_with_replacement(int* perm, int len, int size) {
	int i;
	for (i=0; i<size; ++i)
		perm[i] = rand()%len;
}

void weighted_sample() {
}


// sort index array by values
int gpartition(int* indices, const double* values, int start_index, int end_index) {
	int length = end_index-start_index+1;
	int pidx = start_index + rand()%length;
	double pval = values[indices[pidx]];
	swap(&indices[pidx], &indices[end_index]);

	int i, j=start_index;
	for (i=start_index; i<end_index; ++i) {
		if (values[indices[i]] < pval) {
			swap(&indices[i], &indices[j]);
			j++;
		}
	}

	swap(&indices[j], &indices[end_index]);
	return j;
}

void gsort(int* indices, const double* values, int start_index, int end_index) {
	if (start_index >= end_index)
		return;
	int pidx = gpartition(indices, values, start_index, end_index);
	gsort(indices, values, start_index, pidx-1);
	gsort(indices, values, pidx+1, end_index);
}

int cmp_int(const void* a, const void* b) {
	return *(int*)a - *(int*)b;
}

// labels is assumed to be sorted
int unique(int* labels, int len) {
	if(len<=0) return 0;
	else if(len==1) return 1;

	int i, c=1;
	for (i=1; i<len; ++i) {
		if (labels[i] != labels[i-1])
			c += 1;
	}
	return c;
}

#endif /* _UTIL_H */
