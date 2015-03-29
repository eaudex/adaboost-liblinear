#ifndef _ADABOOST_LINEAR_H
#define _ADABOOST_LINEAR_H
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>
#include "util.h"
#include "linear.h"
#include "problem.h"

struct adaboost_linear_parameter {
	int max_iter;
	double sample_rate;
	struct parameter* linear_param;
};
void print_linear_parameter(const struct parameter* param) {
	printf("[base_parameter] solver_type %d reg_param %g stop_tol %g\n",
			param->solver_type, param->C, param->eps);
}
void print_adaboost_linear_parameter(const struct adaboost_linear_parameter* param) {
	printf("[meta_parameter] max_iter %d sample_rate %g\n", param->max_iter, param->sample_rate);
	print_linear_parameter(param->linear_param);
}
const char* check_adaboost_linear_input(const struct problem* prob, const struct adaboost_linear_parameter* param) {
	if (prob->l <= 0)
		return "l <= 0";
	if (prob->n <= 0)
		return "n <= 0";
	if (param->max_iter <= 0)
		return "max_iter <= 0";
	if (param->sample_rate <= 0.0)
		return "sample_rate <= 0.0";
	if (param->sample_rate>1.0 && param->sample_rate<DBL_MAX)
		return "sample_rate > 1.0";
	return NULL;
}

// TODO
struct adaboost_linear_model {
	int bag_size;
	double* alpha_bag;
	struct model** model_bag;
	double* labels;
};


double bag_predict_instance_label(const struct feature_node* x, const double* alpha_bag, struct model* const* linear_bag, int bag_size) {
	int b;
	double aggr_label = 0.0;
	for (b=0; b<bag_size; ++b) {
		double alpha = alpha_bag[b];
		double label = predict(linear_bag[b], x);
		aggr_label += alpha*label;
	}
	return (aggr_label>0.0)?(linear_bag[0]->label[0]):(linear_bag[0]->label[1]);
}
double bag_predict_labels(const struct problem* prob, const double* alpha_bag, struct model* const* linear_bag, int bag_size, double* aggr_labels) {
	int i;
	double error = 0.0;
	for (i=0; i<prob->l; ++i) {
		double pred_lab = bag_predict_instance_label(prob->x[i], alpha_bag, linear_bag, bag_size);
		if (prob->y[i]*pred_lab < 0.0)
			error += 1.0;
		aggr_labels[i] = pred_lab;
	}
	return error/prob->l;
}


void train_adaboost_linear(const struct problem* prob, const struct adaboost_linear_parameter* param, double** alpha_bag_ret, struct model*** linear_bag_ret, int* bag_size_ret) {
	int max_iter = param->max_iter;
	double sample_rate = param->sample_rate;
	if (max_iter >= INT_MAX)
		max_iter = (int)sqrt(prob->l);
	if (sample_rate*prob->l < 1.0) {
		sample_rate = 1.0;
		fprintf(stderr, "WARNING: sample_size < 1. Set sample_rate to 1.0 (i.e. sample_size = #instances)\n");
	}

	int sample_size = (int)(sample_rate*prob->l);
	if (sample_rate <= 1.0)
		printf("[sample_size] %d/%d\n", sample_size,prob->l);

	// init alpha_bag & model_bag
	int i;
	double* alpha_bag = Malloc(double, max_iter);
	struct model** model_bag = Malloc(struct model*, max_iter);
	for (i=0; i<max_iter; ++i) {
		alpha_bag[i] = 0.0;
		model_bag[i] = NULL;
	}

	int iter = 0;
	double* pred_labels = Malloc(double, prob->l);
	while (iter < max_iter) {
		// train weak learner
		struct model* weak_model = NULL;
		if (sample_rate <= 1.0) {
			//XXX with sampling
			struct problem subprob;
			random_sample(prob, sample_size, &subprob);
			weak_model = train(&subprob, param->linear_param);
			destroy_problem(&subprob);
		}
		else {
			// without sampling
			weak_model = train(prob, param->linear_param);
		}

		// estimate training error
		double error = 0.0;
		for (i=0; i<prob->l; ++i) {
			pred_labels[i] = predict(weak_model, prob->x[i]);
			if (pred_labels[i] != prob->y[i])
				error += prob->W[i];
		}

		// compute alpha
		double score = sqrt((1.0-error)/error);
		double alpha = log(score);
		alpha_bag[iter] = alpha;
		model_bag[iter] = weak_model;

		// update instance weights
		double z = 0.0;
		for (i=0; i<prob->l; ++i) {
			if (pred_labels[i] == prob->y[i])
				prob->W[i] /= score;
			else
				prob->W[i] *= score;
			z += prob->W[i];
		}
		for (i=0; i<prob->l; ++i)
			prob->W[i] /= z;

		//double train_error = bag_predict_labels(prob, alpha_bag, model_bag, iter+1, pred_labels);
		//printf("[%d] error %lf weight_factor %lf alpha %lf train_error %lf\n", iter,error,score,alpha,train_error);
		printf("[%d] error %lf weight_factor %lf alpha %lf\n", iter,error,score,alpha);
		iter++;
	}
	*alpha_bag_ret = alpha_bag;
	*linear_bag_ret = model_bag;
	*bag_size_ret = iter;

	free(pred_labels);
}

void cross_validate_adaboost_linear(const struct problem* prob, const struct adaboost_linear_parameter* param, int nr_fold, double* target_labels) {
	const int l = prob->l;
	const int n = prob->n;
	if (nr_fold > l) {
		nr_fold = l;
		fprintf(stderr, "WARNING: #folds > #instances. Set #folds to #instances instead (i.e. leave-one-out cross validation)\n");
	}

	int i;
	int* perm = Malloc(int, l);
	for(i=0;i<l;++i) perm[i]=i;
	permute(perm, l);

	int* indices = Malloc(int, nr_fold+1);
	for (i=0; i<=nr_fold; ++i)
		indices[i] = i*l/nr_fold;

	for (i=0; i<nr_fold; ++i) {
		int start_index = indices[i];
		int end_index = indices[i+1];
		int fold_size = end_index-start_index;

		// construct training sub-problem
		struct problem subprob;
		subprob.l = l-fold_size;
		subprob.n = n;
		subprob.bias = prob->bias;
		subprob.x = Malloc(struct feature_node*, subprob.l);
		subprob.y = Malloc(double, subprob.l);
		subprob.W = Malloc(double, subprob.l);

		int j, k=0;
		for (j=0; j<start_index; ++j) {
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			//subprob.W[k] = prob->W[perm[j]];
			subprob.W[k] = 1.0/subprob.l;
			//printf("y %g x->index %d x->value %g w %g\n", subprob.y[k],subprob.x[k]->index,subprob.x[k]->value,subprob.W[k]);
			k++;
		}
		for (j=end_index; j<l; ++j) {
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			//subprob.W[k] = prob->W[perm[j]];
			subprob.W[k] = 1.0/subprob.l;
			//printf("y %g x->index %d x->value %g w %g\n", subprob.y[k],subprob.x[k]->index,subprob.x[k]->value,subprob.W[k]);
			k++;
		}

		// train
		int bag_size;
		double* alpha_bag;
		struct model** linear_bag;
		train_adaboost_linear(&subprob, param, &alpha_bag, &linear_bag, &bag_size);

		// test
		double test_error = 0.0;
		for (j=start_index; j<end_index; ++j) {
			target_labels[perm[j]] = bag_predict_instance_label(prob->x[perm[j]], alpha_bag, linear_bag, bag_size);
			if (target_labels[perm[j]] != prob->y[perm[j]])
				test_error += 1.0;
		}
		printf("[fold-%d] (%d-%d) error %g (%d/%d)\n", i,start_index,end_index, test_error/fold_size, (int)test_error,fold_size);

		// free models
		for (j=0; j<bag_size; ++j)
			free_and_destroy_model(&linear_bag[j]);
		free(alpha_bag);
		free(linear_bag);

		// free training sub-problem
		//free(subprob.x);
		//free(subprob.y);
		//free(subprob.W);
		destroy_problem(&subprob);
	}
	free(indices);
	free(perm);
}

int save_alphas(const char* alpha_file_name, const double* alpha_bag, int bag_size) {
	FILE* fp = fopen(alpha_file_name,"w");
	if(fp==NULL) return -1;

	int i;
	fprintf(fp, "%d\n", bag_size);
	for (i=0; i<bag_size; ++i)
		fprintf(fp, "%.16g\n", alpha_bag[i]);

	if (ferror(fp)!=0 || fclose(fp)!=0)
		return -1;
	else
		return 0;
}

int save_bag_linears(const char* model_folder_name, const double* alpha_bag, struct model* const* linear_bag, int bag_size) {
	int status;
	char file_name[1024];

	// make folder
	/*
	struct stat s = {0};
	if (stat(model_folder_name,&s) != -1) {
		fprintf(stderr, "can't make folder %s\n", model_folder_name);
		return -1;
	}
	*/
	status = mkdir(model_folder_name, 0700);
	if (status != 0) {
		fprintf(stderr, "can't make folder %s\n", model_folder_name);
		return -1;
	}

	// save alpha_bag in model_folder
	sprintf(file_name, "%s/alpha", model_folder_name);
	//printf("[alpha_file] %s\n", file_name);
	if (save_alphas(file_name, alpha_bag, bag_size)) {
		fprintf(stderr, "can't save alpha to file %s\n", file_name);
		return -1;
	}

	// save linear_bag in model_folder
	int i;
	for (i=0; i<bag_size; ++i) {
		sprintf(file_name, "%s/%d", model_folder_name,i);
		//printf("[model_file] %s\n", file_name);
		if (save_model(file_name, linear_bag[i])) {
			fprintf(stderr, "can't save model to file %s\n", file_name);
			return -1;
		}
	}

	return 0;
}

int load_bag_linears(const char* model_folder_name, double** alpha_bag_ret, struct model*** linear_bag_ret, int* bag_size_ret) {
	int bag_size = 0;
	double* alpha_bag = NULL;
	struct model** linear_bag = NULL;

	// check model folder
	struct stat s = {0};
	if (stat(model_folder_name,&s) != 0) {
		fprintf(stderr, "can't find model folder %s\n", model_folder_name);
		return -1;
	}

	int i;
	char file_name[1024];
	sprintf(file_name, "%s/alpha", model_folder_name);
	//printf("[alpha_file] %s\n", file_name);
	FILE* fp = fopen(file_name,"r");
	if(fp==NULL) return -1;

	// init models
	fscanf(fp, "%d",&bag_size);
	alpha_bag = Malloc(double, bag_size);
	linear_bag = Malloc(struct model*, bag_size);

	for (i=0; i<bag_size; ++i) {
		fscanf(fp, "%lf", &alpha_bag[i]);
		//printf("%d %.16g\n", i,alpha_bag[i]);
	}
	if (ferror(fp)!=0 || fclose(fp)!=0) {
		fprintf(stderr, "can't load alpha %s\n", file_name);
		return -1;
	}

	for (i=0; i<bag_size; ++i) {
		sprintf(file_name, "%s/%d", model_folder_name,i);
		//printf("[model_file] %s\n", file_name);
		if((linear_bag[i]=load_model(file_name)) == NULL) {
			fprintf(stderr, "can't load lienar model %s\n", file_name);
			return -1;
		}
	}
	*bag_size_ret = bag_size;
	*alpha_bag_ret = alpha_bag;
	*linear_bag_ret = linear_bag;

	return 0;
}


#endif /* _ADABOOST_LINEAR_H */
