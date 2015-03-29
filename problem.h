#ifndef _PROBLEM_H
#define _PROBLEM_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <string.h>
#include "util.h"
#include "linear.h"

struct problem_class {
	int l, n;
	int space_size;
	struct problem prob;
	struct feature_node* x_space;
};
void print_problem_stats(const struct problem_class* prob_cls) {
	printf("[problem] l %d n %d |xspace| %d bias %g\n", prob_cls->l, prob_cls->n, prob_cls->space_size, prob_cls->prob.bias);
}

static void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char* line = NULL;
static int max_line_len;

static char* readline(FILE* input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

// read in a problem (in libsvm format)
int read_problem(const char* filename, struct feature_node** x_space_ret, struct problem* prob, double bias)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	struct feature_node *x_space;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob->l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob->l++;
	}
	rewind(fp);

	prob->bias=bias;

	prob->y = Malloc(double,prob->l);
	prob->x = Malloc(struct feature_node *,prob->l);
	prob->W = Malloc(double,prob->l);
	x_space = Malloc(struct feature_node,elements+prob->l);

	max_index = 0;
	j=0;
	for(i=0;i<prob->l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob->x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob->y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob->bias >= 0)
			x_space[j++].value = prob->bias;

		x_space[j++].index = -1;
	}

	if(prob->bias >= 0)
	{
		prob->n=max_index+1;
		for(i=1;i<prob->l;i++)
			(prob->x[i]-2)->index = prob->n;
		x_space[j-2].index = prob->n;
	}
	else
		prob->n=max_index;

	fclose(fp);

	for(i=0;i<prob->l;i++)
		prob->W[i] = 1.0/prob->l;

	*x_space_ret = x_space;
	return j;
}

// transpose matrix X from row format to column format
void transpose_problem(const struct problem* prob, struct feature_node** x_space_ret, struct problem* prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	long int nnz = 0;
	long int *col_ptr = Malloc(long int,n+1);	//new long int [n+1];
	struct feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = Malloc(double,l);	//new double[l];
	prob_col->x = Malloc(struct feature_node*,n);	//new feature_node*[n];
	prob_col->bias = prob->bias;
	prob_col->W = Malloc(double,l);	//new double[l];

	for(i=0; i<l; i++)
	{
		prob_col->y[i] = prob->y[i];
		prob_col->W[i] = prob->W[i];
	}

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		struct feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = Malloc(struct feature_node,nnz+n);	//new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		struct feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	free(col_ptr);
}


// read in a problem class (in libsvm format)
void read_problem_class(const char* filename, struct problem_class* prob_cls, double bias)
{
	int space_size = read_problem(filename, &prob_cls->x_space, &prob_cls->prob, bias);
	prob_cls->l = prob_cls->prob.l;
	prob_cls->n = prob_cls->prob.n;
	prob_cls->space_size = space_size;
}

// transpose matrix X from row format to column format
void transpose_problem_class(const struct problem_class* prob_cls, struct problem_class* prob_cls_col) {
	transpose_problem(&prob_cls->prob, &prob_cls_col->x_space, &prob_cls_col->prob);
	prob_cls_col->l = prob_cls->prob.l;
	prob_cls_col->n = prob_cls->prob.n;
	prob_cls_col->space_size = prob_cls->space_size;
}


void destroy_problem(struct problem* const prob) {
	free(prob->y);
	free(prob->x);
	free(prob->W);
}
void destroy_problem_class(struct problem_class* const prob_cls) {
	//free(prob_cls->prob.y);
	//free(prob_cls->prob.x);
	//free(prob_cls->prob.W);
	destroy_problem(&prob_cls->prob);
	free(prob_cls->x_space);
}


// without replacement
void random_sample(const struct problem* prob, int size, struct problem* subprob) {
	const int l = prob->l;
	if (size > l) {
		size = l;
		fprintf(stderr, "WARNING: sample_size > #instances. Set sample_size to #instances.\n");
	}

	int i;
	int* perm = Malloc(int, l);
	//for(i=0;i<l;++i) perm[i]=i;
	//sample_without_replacement(perm,l, size);
	sample_with_replacement(perm,l, size);

	subprob->l = size;
	subprob->n = prob->n;
	subprob->bias = prob->bias;
	subprob->x = Malloc(struct feature_node*, subprob->l);
	subprob->y = Malloc(double, subprob->l);
	subprob->W = Malloc(double, subprob->l);

	int k = 0;
	for (i=0; i<size; ++i) {
		subprob->x[k] = prob->x[perm[i]];
		subprob->y[k] = prob->y[perm[i]];
		subprob->W[k] = prob->W[perm[i]];
		k++;
	}
	free(perm);
}


#endif /* _PROBLEM_H */
