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

enum { ADABOOST, ADABOOST_SAMME, ADABOOST_OAA };	//adaboost_solver_type
const char* adaboost_solver_type_table[] =
{
	"ADABOOST",
	"ADABOOST_SAMME",
	"ADABOOST_OAA",
	NULL
};
//XXX copied from `linear.cpp`
const char* base_solver_type_table[] =
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"", "", "",
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};


struct adaboost_linear_parameter {
	int solver_type;
	int max_iter;
	struct parameter* linear_param;
	double sample_rate; //XXX to be removed
};
void print_linear_parameter(const struct parameter* param) {
	printf("[base_parameter] solver_type %s reg_param %g stop_tol %g\n",
			base_solver_type_table[param->solver_type], param->C, param->eps);
}
void print_adaboost_linear_parameter(const struct adaboost_linear_parameter* param) {
	printf("[meta_parameter] solver_type %s max_iter %d sample_rate %g\n",
			adaboost_solver_type_table[param->solver_type], param->max_iter, param->sample_rate);
	print_linear_parameter(param->linear_param);
}
const char* check_adaboost_linear_input(const struct problem_class* prob_cls, const struct adaboost_linear_parameter* param) {
	// check problem
	if (prob_cls->l <= 0)
		return "l <= 0";
	if (prob_cls->n <= 0)
		return "n <= 0";
	if (prob_cls->k < 2)
		return "k < 2";

	// check parameter
	if (param->solver_type != ADABOOST
		&& param->solver_type != ADABOOST_SAMME
		&& param->solver_type != ADABOOST_OAA)
		return "unknown solver type";
	if (param->max_iter <= 0)
		return "max_iter <= 0";
	if (param->sample_rate <= 0.0)
		return "sample_rate <= 0.0";
	if (param->sample_rate>1.0 && param->sample_rate<DBL_MAX)
		return "sample_rate > 1.0";

	// special check
	if (param->solver_type==ADABOOST && prob_cls->k>2)
		return "k>2; please use other types of solvers";

	return NULL;
}

struct adaboost_linear_model {
	struct adaboost_linear_parameter param;
	int bag_size;
	double* alpha_bag;
	struct model** model_bag;
	// used in OAA
	int nr_class;
	double* labels;
};

void destroy_adaboost_linear_model(const struct adaboost_linear_model* adamodel) {
	int i;
	for (i=0; i<adamodel->bag_size; ++i)
		free_and_destroy_model(&adamodel->model_bag[i]);
	free(adamodel->alpha_bag);
	free(adamodel->model_bag);
	free(adamodel->labels);
}


// use case: nr_class=2, or SAMME
static double bag_predict_instance_label(const struct feature_node* x, const struct adaboost_linear_model* adamodel, double* dec_value) {
	const int bag_size = adamodel->bag_size;
	const double* alpha_bag = adamodel->alpha_bag;

	int b;
	int* indices = Malloc(int,bag_size);
	double* pred_labels = Malloc(double,bag_size);
	for (b=0; b<bag_size; ++b) {
		indices[b] = b;
		pred_labels[b] = predict(adamodel->model_bag[b], x);
	}
	gsort(indices, pred_labels, 0, bag_size-1);

	double aggr_label = pred_labels[indices[0]];
	double alpha_sum = alpha_bag[indices[0]];
	double max_sum = alpha_bag[indices[0]];
	for (b=1; b<bag_size; ++b) {
		if (pred_labels[indices[b]] == pred_labels[indices[b-1]]) {
			alpha_sum += alpha_bag[indices[b]];
		}
		else {
			if (alpha_sum > max_sum) {
				aggr_label = pred_labels[indices[b-1]];
				max_sum = alpha_sum;
			}
			alpha_sum = alpha_bag[indices[b]];
		}
	}
	if (alpha_sum > max_sum) {
		aggr_label = pred_labels[indices[b-1]];
		max_sum = alpha_sum;
	}

	free(pred_labels);
	free(indices);

	*dec_value = max_sum;
	return aggr_label;
}

// one-against-all (OAA) decomposition
// use case: nr_class > 2
static double bag_predict_instance_label_OAA(const struct feature_node* x, const struct adaboost_linear_model* adamodel, double* dec_value) {
	const int bag_size = adamodel->bag_size;
	const int nr_class = adamodel->nr_class;
	const int minibag_size = bag_size/nr_class;

	// init binary adamodel (no need to destroy in the end)
	struct adaboost_linear_model _adamodel;
	_adamodel.bag_size = minibag_size;
	_adamodel.nr_class = 2;
	_adamodel.labels = NULL;
	_adamodel.alpha_bag = &adamodel->alpha_bag[0];
	_adamodel.model_bag = &adamodel->model_bag[0];

	// arg max_k alpha_sum_k
	int k;
	double tmp_dec_value;
	double pred_label = bag_predict_instance_label(x, &_adamodel, &tmp_dec_value);
	double max_dec_value = pred_label*tmp_dec_value;
	int argmax_dec_value = 0;

	for (k=1; k<nr_class; ++k) {
		// re-init
		_adamodel.alpha_bag = &adamodel->alpha_bag[k*minibag_size];
		_adamodel.model_bag = &adamodel->model_bag[k*minibag_size];
		// predict
		pred_label = bag_predict_instance_label(x, &_adamodel, &tmp_dec_value);
		tmp_dec_value *= pred_label;
		if (max_dec_value < tmp_dec_value) {
			max_dec_value = tmp_dec_value;
			argmax_dec_value = k;
		}
	}

	*dec_value = max_dec_value;
	return adamodel->labels[argmax_dec_value];
}

static double bag_predict_labels(const struct problem* prob, const struct adaboost_linear_model* adamodel, double* aggr_labels) {
	int i;
	double dec_value, error=0.0;
	for (i=0; i<prob->l; ++i) {
		double pred_lab = bag_predict_instance_label(prob->x[i], adamodel, &dec_value);
		if (pred_lab != prob->y[i])
			error += 1.0;
		aggr_labels[i] = pred_lab;
	}
	return error/prob->l;
}
static double bag_predict_labels_OAA(const struct problem* prob, const struct adaboost_linear_model* adamodel, double* aggr_labels) {
	int i;
	double dec_value, error=0.0;
	for (i=0; i<prob->l; ++i) {
		double pred_lab = bag_predict_instance_label_OAA(prob->x[i], adamodel, &dec_value);
		if (pred_lab != prob->y[i])
			error += 1.0;
		aggr_labels[i] = pred_lab;
	}
	return error/prob->l;
}

// predict instance
double predict_adaboost_linear(const struct feature_node* x, const struct adaboost_linear_model* adamodel, double* dec_value) {
	if (adamodel->param.solver_type == ADABOOST) {
		return bag_predict_instance_label(x, adamodel, dec_value);
	}
	else if (adamodel->param.solver_type == ADABOOST_SAMME) {
		return bag_predict_instance_label(x, adamodel, dec_value);
	}
	else if (adamodel->param.solver_type == ADABOOST_OAA) {
		return bag_predict_instance_label_OAA(x, adamodel, dec_value);
	}
	else {
		return -1.0;
	}
}
// predict problem
double predict_adaboost_linear(const struct problem* prob, const struct adaboost_linear_model* adamodel, double* aggr_labels) {
	if (adamodel->param.solver_type == ADABOOST) {
		return bag_predict_labels(prob, adamodel, aggr_labels);
	}
	else if (adamodel->param.solver_type == ADABOOST_SAMME) {
		return bag_predict_labels(prob, adamodel, aggr_labels);
	}
	else if (adamodel->param.solver_type == ADABOOST_OAA) {
		return bag_predict_labels_OAA(prob, adamodel, aggr_labels);
	}
	else {
		return -1.0;
	}
}


// Standard AdaBoost: binary classification only
//TODO remove sampling...
void train_adaboost_linear_binary(const struct problem_class* prob_cls, const struct adaboost_linear_parameter* param, struct adaboost_linear_model* adamodel) {
	int max_iter = param->max_iter;
	double sample_rate = param->sample_rate;
	if (max_iter >= INT_MAX)
		max_iter = (int)sqrt(prob_cls->l);
	if (sample_rate*prob_cls->l < 1.0) {
		sample_rate = 1.0;
		fprintf(stderr, "WARNING: sample_size < 1. Set sample_rate to 1.0 (i.e. sample_size = #instances)\n");
	}

	const struct problem* prob = &prob_cls->prob;

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
	adamodel->bag_size = iter;
	adamodel->alpha_bag = alpha_bag;
	adamodel->model_bag = model_bag;
	//adamodel->alpha_bag = (double*) realloc((void*)alpha_bag, sizeof(double)*iter);
	//adamodel->model_bag = (struct model**) realloc((void*)model_bag, sizeof(struct model*)*iter);
	adamodel->nr_class = 2;
	adamodel->labels = NULL;

	free(pred_labels);
}

// one-against-all decomposition
// prob_cls->prob.y points to a tmp label array, and points back to the original label array in the end
void train_adaboost_linear_OAA(struct problem_class* const prob_cls, const struct adaboost_linear_parameter* param, struct adaboost_linear_model* adamodel) {
	const int l = prob_cls->l;
	const int n = prob_cls->n;
	const int nr_class = prob_cls->k;
	const int max_iter = param->max_iter;

	// init adamodel
	adamodel->nr_class = nr_class;
	adamodel->labels = Malloc(double, nr_class);
	adamodel->bag_size = max_iter*nr_class;	//XXX
	adamodel->alpha_bag = Malloc(double, max_iter*nr_class);
	adamodel->model_bag = Malloc(struct model*, max_iter*nr_class);

	double* prob_y = prob_cls->prob.y;	//original labels
	double* y2 = Malloc(double,l);		//binary labels
	prob_cls->prob.y = y2;				//point to binary labels

	int i, k;
	for (k=0; k<nr_class; ++k) {
		double yk = prob_cls->y_space[k];
		for (i=0; i<l; ++i) {
			if (prob_y[i] == yk)
				y2[i] = +1.0;
			else
				y2[i] = -1.0;
		}
		// ....
		init_instance_weight(prob_cls);	//TODO follows AdaBoost.MH setting
		// ....

		// train binary adamodel
		struct adaboost_linear_model _adamodel;
		train_adaboost_linear_binary(prob_cls, param, &_adamodel);

		// copy binary adamodel content
		adamodel->labels[k] = yk;
		int offset = k*max_iter;
		for (i=0; i<max_iter; ++i) {
			adamodel->alpha_bag[offset+i] = _adamodel.alpha_bag[i];
			adamodel->model_bag[offset+i] = _adamodel.model_bag[i];
		}
		//XXX cannot free linear models	//destroy_adaboost_linear_model(&_adamodel);
		free(_adamodel.alpha_bag);
		free(_adamodel.model_bag);
		free(_adamodel.labels);
	}
	free(y2);
	prob_cls->prob.y = prob_y;	//point back to original labels

	//evaluate performance below
	double* pred_labels = Malloc(double,l);
	double train_error = bag_predict_labels_OAA(&prob_cls->prob, adamodel, pred_labels);
	printf("train_error %lf\n", train_error);
	free(pred_labels);
}

// SAMME: multi-class classification
// reduces to standard AdaBoost in binary classification
void train_adaboost_linear_samme(const struct problem_class* prob_cls, const struct adaboost_linear_parameter* param, struct adaboost_linear_model* adamodel) {
	int max_iter = param->max_iter;
	double sample_rate = param->sample_rate;
	if (max_iter >= INT_MAX)
		max_iter = (int)sqrt(prob_cls->l);
	if (sample_rate*prob_cls->l < 1.0) {
		sample_rate = 1.0;
		fprintf(stderr, "WARNING: sample_size < 1. Set sample_rate to 1.0 (i.e. sample_size = #instances)\n");
	}

	const struct problem* prob = &prob_cls->prob;

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
		double score = (double)(prob_cls->k-1)*(1.0-error)/error;
		double alpha = log(score);
		alpha_bag[iter] = alpha;
		model_bag[iter] = weak_model;

		// update instance weights
		double z = 0.0;
		for (i=0; i<prob->l; ++i) {
			if (pred_labels[i] != prob->y[i])
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
	adamodel->bag_size = iter;
	adamodel->alpha_bag = alpha_bag;
	adamodel->model_bag = model_bag;
	//adamodel->alpha_bag = (double*) realloc((void*)alpha_bag, sizeof(double)*iter);
	//adamodel->model_bag = (struct model**) realloc((void*)model_bag, sizeof(struct model*)*iter);
	adamodel->nr_class = prob_cls->k;
	adamodel->labels = NULL;

	free(pred_labels);
}

void train_adaboost_linear(struct problem_class* prob_cls, const struct adaboost_linear_parameter* param, struct adaboost_linear_model* adamodel) {
	adamodel->param = *param;

	if (param->solver_type == ADABOOST) {
		train_adaboost_linear_binary(prob_cls, param, adamodel);
	}
	else if (param->solver_type == ADABOOST_SAMME) {
		train_adaboost_linear_samme(prob_cls, param, adamodel);
	}
	else if (param->solver_type == ADABOOST_OAA) {
		train_adaboost_linear_OAA(prob_cls, param, adamodel);
	}
	else {
	}
}

void cross_validate_adaboost_linear(const struct problem_class* prob_cls, const struct adaboost_linear_parameter* param, int nr_fold, double* target_labels) {
	const int l = prob_cls->l;
	const int n = prob_cls->n;
	const int k = prob_cls->k;
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
		struct problem_class subprob_cls;
		subprob_cls.l = l-fold_size;
		subprob_cls.n = n;
		subprob_cls.k = k;
		// init subprob_cls.prob ...
		subprob_cls.prob.l = subprob_cls.l;
		subprob_cls.prob.n = subprob_cls.n;
		subprob_cls.prob.bias = prob_cls->prob.bias;
		subprob_cls.prob.x = Malloc(struct feature_node*, subprob_cls.l);
		subprob_cls.prob.y = Malloc(double, subprob_cls.l);
		subprob_cls.prob.W = Malloc(double, subprob_cls.l);
		// end of init
		subprob_cls.space_size = 0;
		subprob_cls.x_space = NULL;
		subprob_cls.y_space = Malloc(double, subprob_cls.k);

		int j, c=0;
		for (j=0; j<start_index; ++j) {
			subprob_cls.prob.x[c] = prob_cls->prob.x[perm[j]];
			subprob_cls.prob.y[c] = prob_cls->prob.y[perm[j]];
			subprob_cls.prob.W[c] = 1.0/subprob_cls.l;
			//printf("y %g x->index %d x->value %g w %g\n", subprob_cls.prob.y[c],subprob_cls.prob.x[c]->index,subprob_cls.prob.x[c]->value,subprob_cls.prob.W[c]);
			c++;
		}
		for (j=end_index; j<l; ++j) {
			subprob_cls.prob.x[c] = prob_cls->prob.x[perm[j]];
			subprob_cls.prob.y[c] = prob_cls->prob.y[perm[j]];
			subprob_cls.prob.W[c] = 1.0/subprob_cls.l;
			//printf("y %g x->index %d x->value %g w %g\n", subprob_cls.prob.y[c],subprob_cls.prob.x[c]->index,subprob_cls.prob.x[c]->value,subprob_cls.prob.W[c]);
			c++;
		}
		for (c=0; c<k; ++c)
			subprob_cls.y_space[c] = prob_cls->y_space[c];

		// train
		struct adaboost_linear_model adamodel;
		train_adaboost_linear(&subprob_cls, param, &adamodel);

		// test
		double dec_value, test_error=0.0;
		for (j=start_index; j<end_index; ++j) {
			target_labels[perm[j]] = predict_adaboost_linear(prob_cls->prob.x[perm[j]], &adamodel, &dec_value);
			if (target_labels[perm[j]] != prob_cls->prob.y[perm[j]])
				test_error += 1.0;
		}
		printf("[fold-%d] (%d-%d) error %g (%d/%d)\n", i,start_index,end_index, test_error/fold_size, (int)test_error,fold_size);

		// free model
		destroy_adaboost_linear_model(&adamodel);

		// free sub-problem
		destroy_problem_class(&subprob_cls);
	}
	free(indices);
	free(perm);
}


//int save_alpha(const char* alpha_file_name, const double* alpha_bag, int bag_size) {
static int save_alpha(const char* alpha_file_name, const struct adaboost_linear_model* adamodel) {
	FILE* fp = fopen(alpha_file_name,"w");
	if(fp==NULL) return -1;

	fprintf(fp, "solver_type %s\n", adaboost_solver_type_table[adamodel->param.solver_type]);
	fprintf(fp, "max_iter %d\n", adamodel->param.max_iter);
	fprintf(fp, "nr_class %d\n", adamodel->nr_class);

	int i;
	if (adamodel->labels != NULL) {
		fprintf(fp, "labels %lf", adamodel->labels[0]);
		for (i=1; i<adamodel->nr_class; ++i)
			fprintf(fp, " %lf", adamodel->labels[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "alpha %d\n", adamodel->bag_size);
	for (i=0; i<adamodel->bag_size; ++i)
		fprintf(fp, "%.16g\n", adamodel->alpha_bag[i]);

	if (ferror(fp)!=0 || fclose(fp)!=0)
		return -1;
	else
		return 0;
}

int save_adaboost_linear_model(const char* model_folder_name, const struct adaboost_linear_model* adamodel) {
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
	if (save_alpha(file_name, adamodel)) {
		fprintf(stderr, "can't save alpha to file %s\n", file_name);
		return -1;
	}

	// save model_bag in model_folder
	int i;
	for (i=0; i<adamodel->bag_size; ++i) {
		sprintf(file_name, "%s/%d", model_folder_name,i);
		if (save_model(file_name, adamodel->model_bag[i])) {
			fprintf(stderr, "can't save model to file %s\n", file_name);
			return -1;
		}
	}

	return 0;
}

int load_adaboost_linear_model(const char* model_folder_name, struct adaboost_linear_model* adamodel) {
	// model parameters
	int solver_type = -1;
	int max_iter = 0;
	int bag_size = 0;
	double* alpha_bag = NULL;
	struct model** model_bag = NULL;
	int nr_class = 0;
	double* labels = NULL;

	// check model folder
	struct stat s = {0};
	if (stat(model_folder_name,&s) != 0) {
		fprintf(stderr, "can't find model folder %s\n", model_folder_name);
		return -1;
	}

	// init models
	int i;
	char file_name[1024];
	sprintf(file_name, "%s/alpha", model_folder_name);
	FILE* fp = fopen(file_name,"r");
	if (fp == NULL) {
		fprintf(stderr, "can't open model file %s\n", file_name);
		return -1;
	}

	char str[32];
	while (1) {
		fscanf(fp, "%s", str);
		if (strcmp(str,"solver_type") == 0) {
			fscanf(fp, "%s", str);
			for (i=0; adaboost_solver_type_table[i]; ++i) {
				if (strcmp(str,adaboost_solver_type_table[i]) == 0) {
					solver_type = i;
					break;
				}
			}
			if (adaboost_solver_type_table[i] == NULL) {
				fprintf(stderr, "unknown solver type %s\n", str);
				free(labels);
				return -1;
			}
			//printf("solver_type %s\n", str);
		}
		else if (strcmp(str,"max_iter") == 0) {
			fscanf(fp, "%d", &max_iter);
			//printf("max_iter %d\n", max_iter);
		}
		else if (strcmp(str,"nr_class") == 0) {
			fscanf(fp, "%d", &nr_class);
			//printf("nr_class %d\n", nr_class);
		}
		else if (strcmp(str,"labels") == 0) {
			int i;
			labels = Malloc(double,nr_class);
			//printf("labels");
			for (i=0; i<nr_class; ++i) {
				fscanf(fp, "%lf", &labels[i]);
				//printf(" %g", labels[i]);
			}
			//printf("\n");
		}
		else if (strcmp(str,"alpha") == 0) {
			fscanf(fp, "%d", &bag_size);
			//printf("bag_size %d\n", bag_size);
			if (bag_size <= 0) {
				fprintf(stderr, "model size <= 0: %d\n", bag_size);
				return -1;
			}
			break;
		}
		else {
			fprintf(stderr, "unknown format in %s: %s\n", file_name,str);
			free(labels);
			return -1;
		}
	}

	// load alpha values
	alpha_bag = Malloc(double, bag_size);
	for (i=0; i<bag_size; ++i)
		fscanf(fp, "%lf", &alpha_bag[i]);
	if (ferror(fp)!=0 || fclose(fp)!=0) {
		fprintf(stderr, "can't load alpha %s\n", file_name);
		return -1;
	}

	// load linear models
	model_bag = Malloc(struct model*, bag_size);
	for (i=0; i<bag_size; ++i) {
		sprintf(file_name, "%s/%d", model_folder_name,i);
		if((model_bag[i]=load_model(file_name)) == NULL) {
			fprintf(stderr, "can't load lienar model %s\n", file_name);
			return -1;
		}
	}
	adamodel->param.solver_type = solver_type;
	adamodel->param.max_iter = max_iter;
	adamodel->bag_size = bag_size;
	adamodel->alpha_bag = alpha_bag;
	adamodel->model_bag = model_bag;
	adamodel->nr_class = nr_class;
	adamodel->labels = labels;

	return 0;
}



#endif /* _ADABOOST_LINEAR_H */
