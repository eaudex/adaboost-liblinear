#ifndef _STUMP_H
#define _STUMP_H

#include <limits.h>
#include <float.h>
#include "util.h"
#include "math.h"

extern int quiet;

struct adaboost_stump_parameter {
	int max_iter;
	int random_subspace;
};
void print_adaboost_stump_parameter(const struct adaboost_stump_parameter* param) {
	printf("[parameter] max_iter %d random_subspace %d\n", param->max_iter, param->random_subspace);
}
const char* check_adaboost_input(const struct problem_class* prob_cls, const struct adaboost_stump_parameter* param) {
	if (prob_cls->l <= 0)
		return "l <= 0";
	if (prob_cls->n <= 0)
		return "n <= 0";
	if (param->max_iter <= 0)
		return "max_iter <= 0";
	if (param->random_subspace <= 0)
		return "random_subspace <= 0";
	return NULL;
}

struct stump {
	int dimension;		//feature index
	int direction;		//+1:+1 lable on the right-hand side of [position]; -1:-1 label on the right-hand side of [position]
	double position;	//threshold
	double rate;		//instance-weighted training error
	double weight;		//model weight, i.e. alpha value in Adaboost
};
void print_stump(const struct stump* stump_) {
	printf("[stump] feature %d direction %d position %g rate %g weight %g\n",
			stump_->dimension, stump_->direction, stump_->position, stump_->rate, stump_->weight);
}

int save_bag_stumps(const char* model_file_name, struct stump* const* stump_bag, int bag_size) {
	FILE* fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	fprintf(fp, "%d\n", bag_size);
	int b;
	for (b=0; b<bag_size; ++b)
		fprintf(fp, "%d %d %.16g %.16g %.16g\n",
			stump_bag[b]->dimension, stump_bag[b]->direction, stump_bag[b]->position, stump_bag[b]->rate, stump_bag[b]->weight);

	if (ferror(fp)!=0 || fclose(fp)!=0)
		return -1;
	else
		return 0;
}
int load_bag_stumps(const char* model_file_name, struct stump*** const stump_bag_ret, int* const bag_size_ret) {
	FILE* fp = fopen(model_file_name,"r");
	if(fp==NULL) return -1;

	int b, num_stumps;
	fscanf(fp, "%d", &num_stumps);

	struct stump** stumps_ = Malloc(struct stump*,num_stumps);
	for (b=0; b<num_stumps; ++b)
		stumps_[b] = Malloc(struct stump,1);

	for (b=0; b<num_stumps; ++b) {
		fscanf(fp, "%d", &(stumps_[b]->dimension));
		fscanf(fp, "%d", &(stumps_[b]->direction));
		fscanf(fp, "%lf", &(stumps_[b]->position));
		fscanf(fp, "%lf", &(stumps_[b]->rate));
		fscanf(fp, "%lf", &(stumps_[b]->weight));
	}

	*bag_size_ret = num_stumps;
	*stump_bag_ret = stumps_;

	if (ferror(fp)!=0 || fclose(fp)!=0)
		return -1;
	else
		return 0;
}

double predict_value(double feature_value, const struct stump* stump_) {
	return (feature_value>stump_->position)?(1.0*stump_->direction):(-1.0*stump_->direction);
}
double predict_labels(const struct problem* prob_col, const struct stump* stump_, double* pred_labels) {
	const int feature_index = stump_->dimension;
	const double pred_lab_at0 = predict_value(0.0, stump_);
	memset((void*)pred_labels, 0, sizeof(double)*prob_col->l);

	int i;
	int curr_idx = 0;
	double error = 0.0;
	struct feature_node* xj = prob_col->x[feature_index];
	while (xj->index != -1) {
		int idx = xj->index-1;
		for (i=curr_idx; i<idx; ++i) {
			pred_labels[i] = pred_lab_at0;
			if (prob_col->y[i]*pred_lab_at0 < 0.0)
				error += prob_col->W[i];
		}
		double pred_lab = predict_value(xj->value, stump_);
		pred_labels[idx] = pred_lab;
		if (prob_col->y[idx]*pred_lab < 0.0)
			error += prob_col->W[idx];
		curr_idx = idx+1;
		xj ++;
	}
	for (i=curr_idx; i<prob_col->l; ++i) {
		pred_labels[i] = pred_lab_at0;
		if (prob_col->y[i]*pred_lab_at0 < 0.0)
			error += prob_col->W[i];
	}

	return error;
}

double bag_predict_labels0(const struct problem* prob_col, struct stump* const* stump_bag, int bag_size, double* aggr_labels) {
	memset((void*)aggr_labels, 0, sizeof(double)*prob_col->l);

	int b, i;
	double* pred_labels = Malloc(double,prob_col->l);
	for (b=0; b<bag_size; ++b) {
		double alpha = stump_bag[b]->weight;
		predict_labels(prob_col, stump_bag[b], pred_labels);
		for (i=0; i<prob_col->l; ++i)
			aggr_labels[i] += alpha*pred_labels[i];
	}

	double error = 0.0;
	for (i=0; i<prob_col->l; ++i) {
		double pred_lab  = (aggr_labels[i]>0.0)?(+1.0):(-1.0);
		if (prob_col->y[i]*pred_lab < 0.0)
			error += 1.0;
		aggr_labels[i] = pred_lab;
	}

	free(pred_labels);
	return error/prob_col->l;
}

double predict_instance_label(const struct feature_node* x, const struct stump* stump_) {
	const int feature_index = stump_->dimension+1;
	const struct feature_node* xj = x;
	while (xj->index!=-1 && xj->index<feature_index)
		xj ++;

	if (xj->index == feature_index)
		return predict_value(xj->value, stump_);
	else
		return predict_value(0.0, stump_);
}
double bag_predict_instance_label(const struct feature_node* x, struct stump* const* stump_bag, int bag_size) {
	int b;
	double aggr_label = 0.0;
	for (b=0; b<bag_size; ++b) {
		double alpha = stump_bag[b]->weight;
		double label = predict_instance_label(x, stump_bag[b]);
		aggr_label += alpha*label;
	}
	return (aggr_label>0.0)?(+1.0):(-1.0);
}
double bag_predict_labels(const struct problem* prob, struct stump* const* stump_bag, int bag_size, double* aggr_labels) {
	int i;
	double error = 0.0;
	for (i=0; i<prob->l; ++i) {
		double pred_lab = bag_predict_instance_label(prob->x[i], stump_bag, bag_size);
		if (prob->y[i]*pred_lab < 0.0)
			error += 1.0;
		aggr_labels[i] = pred_lab;
	}
	return error/prob->l;
}


int cmp(const void* a, const void* b) {
	struct feature_node* aa = *(struct feature_node**)a;
	struct feature_node* bb = *(struct feature_node**)b;
	if (aa->value < bb->value)
		return -1;
	else if (aa->value > bb->value)
		return +1;
	else
		return 0;
}

double find_stump1(const struct problem* prob_col, struct feature_node* const* feature_nodes, int length, double sum_pos_weights, double sum_neg_weights, double sum_pos_weights_at0, double sum_neg_weights_at0, struct stump* stump_) {
	// sort feature_nodes
	qsort((void*)feature_nodes, length, sizeof(struct feature_node*), cmp);

	int i;
	int best_dir = +1;
	int best_cut = -1;
	double rate = sum_pos_weights;
	double best_pos;
	double best_rate = sum_pos_weights;
	struct feature_node* xj;
	for (i=0; i<length; ++i) {
		xj = feature_nodes[i];
		int idx = xj->index-1;
		double val = xj->value;

		if (idx < 0) {
			rate -= sum_pos_weights_at0;
			rate += sum_neg_weights_at0;
		}
		else {
			double lab = prob_col->y[idx];
			if (lab > 0.0)
				rate -= prob_col->W[idx];
			else
				rate += prob_col->W[idx];
		}

		if (i<length-1 && val==feature_nodes[i+1]->value)
			continue;
		if (rate > best_rate) {
			best_cut = i;
			best_rate = rate;
		}
	}
	if (best_cut <= -1)
		best_pos = feature_nodes[0]->value-0.01;
	else if (best_cut < length-1)
		best_pos = 0.5*(feature_nodes[best_cut]->value+feature_nodes[best_cut+1]->value);
	else
		best_pos = feature_nodes[length-1]->value+0.01;

	best_cut = -1;
	rate = sum_neg_weights;
	for (i=0; i<length; ++i) {
		xj = feature_nodes[i];
		int idx = xj->index-1;
		double val = xj->value;

		if (idx < 0) {
			rate += sum_pos_weights_at0;
			rate -= sum_neg_weights_at0;
		}
		else {
			double lab = prob_col->y[idx];
			if (lab > 0.0)
				rate += prob_col->W[idx];
			else
				rate -= prob_col->W[idx];
		}

		if (i<length-1 && val==feature_nodes[i+1]->value)
			continue;
		if (rate > best_rate) {
			best_cut = i;
			best_dir = -1;
			best_rate = rate;
		}
	}
	if (best_dir == -1) {
		if (best_cut <= -1)
			best_pos = feature_nodes[0]->value-0.01;
		else if (best_cut < length-1)
			best_pos = 0.5*(feature_nodes[best_cut]->value+feature_nodes[best_cut+1]->value);
		else
			best_pos = feature_nodes[length-1]->value+0.01;
	}

	stump_->direction = best_dir;
	stump_->position = best_pos;
	stump_->rate = 1.0-best_rate;

	return 1.0-best_rate;
}

double find_stump0(const struct problem* prob_col, int feature_index, double sum_pos_weights, double sum_neg_weights, struct stump* stump_) {
	const int l = prob_col->l;
	stump_->dimension = feature_index;

	struct feature_node* node0 = Malloc(struct feature_node,1);
	node0->index = 0;
	node0->value = 0.0;
	struct feature_node** feature_nodes = Malloc(struct feature_node*,l);

	double sum_pos_weights_at0 = 0.0;;
	double sum_neg_weights_at0 = 0.0;;
	struct feature_node* xj = prob_col->x[feature_index];
	int i, curr_idx=0, node_count=0;
	while (xj->index != -1) {
		int idx = xj->index-1;
		feature_nodes[node_count++] = xj;
		for (i=curr_idx; i<idx; ++i) {
			if (prob_col->y[i] > 0.0)
				sum_pos_weights_at0 += prob_col->W[i];
			else
				sum_neg_weights_at0 += prob_col->W[i];
		}
		curr_idx = idx+1;
		xj ++;
	}
	for (i=curr_idx; i<l; ++i) {
		if (prob_col->y[i] > 0.0)
			sum_pos_weights_at0 += prob_col->W[i];
		else
			sum_neg_weights_at0 += prob_col->W[i];
	}
	if (node_count < l) //imply at least one feature_node has zero value
		feature_nodes[node_count++] = node0;

	//printf("[%d] %lf %lf | %d %lf %lf\n", feature_index,sum_pos_weights,sum_neg_weights,node_count,sum_pos_weights_at0,sum_neg_weights_at0);
	double rate = find_stump1(prob_col, feature_nodes, node_count, sum_pos_weights, sum_neg_weights, sum_pos_weights_at0, sum_neg_weights_at0, stump_);

	free(feature_nodes);
	free(node0);
	return rate;
}

double find_best_stump(const struct problem* prob_col, struct stump* stump_, int random_subspace) {
	const int l_size = prob_col->l;
	const int w_size = prob_col->n;

	int i;
	double sum_pos_weights = 0.0;
	double sum_neg_weights = 0.0;
	for (i=0; i<prob_col->l; ++i) {
		if (prob_col->y[i] > 0.0)
			sum_pos_weights += prob_col->W[i];
		else
			sum_neg_weights += prob_col->W[i];
	}
	//printf("pos %lf neg %lf\n", sum_pos_weights,sum_neg_weights);

	int j;
	double min_error = DBL_MAX;
	struct stump* tmp_stump = Malloc(struct stump,1);
	if (random_subspace >= w_size) {
		// deterministic: entire space
		for (j=0; j<w_size; ++j) {
			double error = find_stump0(prob_col, j, sum_pos_weights, sum_neg_weights, tmp_stump);
			//print_stump(tmp_stump);
			if (error < min_error) {
				min_error = error;
				stump_->dimension = j;
				stump_->direction = tmp_stump->direction;
				stump_->position = tmp_stump->position;
				stump_->rate = error;
			}
		}
	}
	else {
		// stochastic: random subspace
		int* indices = Malloc(int,w_size);
		for (j=0; j<w_size; ++j)
			indices[j] = j;
		for (j=0; j<random_subspace; ++j) {
			int jj = j + rand()%(w_size-j);
			swap(&indices[j], &indices[jj]);
		}
		for (j=0; j<random_subspace; ++j) {
			//int rand_j = rand()%w_size;
			int rand_j = indices[j];
			double error = find_stump0(prob_col, rand_j, sum_pos_weights, sum_neg_weights, tmp_stump);
			//print_stump(tmp_stump);
			if (error < min_error) {
				min_error = error;
				stump_->dimension = rand_j;
				stump_->direction = tmp_stump->direction;
				stump_->position = tmp_stump->position;
				stump_->rate = error;
			}
		}
		free(indices);
	}
	stump_->weight = 0.5*log((1.0-min_error)/min_error);

	free(tmp_stump);
	return min_error;
}

void train_adaboost_stump(const struct problem_class* prob_cls, const struct adaboost_stump_parameter* param, struct stump*** stump_bag_ret, int* bag_size_ret) {
	int max_iter = param->max_iter;
	int random_subspace = param->random_subspace;
	if (max_iter >= INT_MAX)
		max_iter = (int)sqrt(prob_cls->l);
	if (random_subspace >= prob_cls->n)
		random_subspace = prob_cls->n;

	// transform data to compressed column format
	struct problem_class prob_cls_col;
	transpose_problem_class(prob_cls, &prob_cls_col);
	struct problem* prob_col = &prob_cls_col.prob;

	int i, iter=0;
	double* pred_labels = Malloc(double, prob_col->l);
	struct stump** stumps = Malloc(struct stump*, max_iter);
	while (iter < max_iter) {
		// select & train weak learner
		stumps[iter] = Malloc(struct stump, 1);
		find_best_stump(prob_col, stumps[iter], random_subspace);
		if(quiet==0) print_stump(stumps[iter]);

		// estimate error rate & compute alpha
		double error = predict_labels(prob_col, stumps[iter], pred_labels);
		double score = sqrt((1.0-error)/error);
		double alpha = log(score);	//==stumps[iter]->weight

		// update instance weights
		double z = 0.0;
		for (i=0; i<prob_col->l; ++i) {
			if (prob_col->y[i]*pred_labels[i] > 0.0)
				prob_col->W[i] /= score;
			else
				prob_col->W[i] *= score;
			z += prob_col->W[i];
		}
		for (i=0; i<prob_col->l; ++i)
			prob_col->W[i] /= z;

		double train_error = bag_predict_labels0(prob_col,stumps,iter+1,pred_labels);
		printf("[%d] error %lf score %lf alpha %lf train_error %lf\n", iter,error,score,alpha,train_error);
		iter ++;
	}
	*stump_bag_ret = stumps;
	*bag_size_ret = iter;

	free(pred_labels);
	destroy_problem_class(&prob_cls_col);
}


void cross_validate_adaboost_stump(const struct problem_class* prob_cls, const struct adaboost_stump_parameter* param, int nr_fold, double* target_labels) {
	if (nr_fold > prob_cls->l) {
		nr_fold = prob_cls->l;
		fprintf(stderr, "WARNING: #folds > #instances. Set #folds to #instances instead (i.e. leave-one-out cross validation)\n");
	}

	int i;
	int* perm = Malloc(int, prob_cls->l);
	for (i=0; i<prob_cls->l; ++i)
		perm[i] = i;
	permute(perm, prob_cls->l);

	int* indices = Malloc(int, nr_fold+1);
	for (i=0; i<=nr_fold; ++i)
		indices[i] = i * prob_cls->l / nr_fold;

	struct problem_class subprob_cls;
	for (i=0; i<nr_fold; ++i) {
		int start_index = indices[i];
		int end_index = indices[i+1];
		int fold_size = end_index-start_index;

		// construct training sub-problem
		subprob_cls.l = prob_cls->l - fold_size;
		subprob_cls.n = prob_cls->n;
		// init subprob_cls.prob ...
		subprob_cls.prob.l = subprob_cls.l;
		subprob_cls.prob.n = subprob_cls.n;
		subprob_cls.prob.y = Malloc(double, subprob_cls.l);
		subprob_cls.prob.x = Malloc(struct feature_node*, subprob_cls.l);
		subprob_cls.prob.bias = prob_cls->prob.bias;
		subprob_cls.prob.W = Malloc(double, subprob_cls.l);
		// end of init
		subprob_cls.space_size = 0;
		subprob_cls.x_space = NULL;

		int j, k=0;
		for (j=0; j<start_index; ++j) {
			subprob_cls.prob.y[k] = prob_cls->prob.y[perm[j]];
			subprob_cls.prob.x[k] = prob_cls->prob.x[perm[j]];
			subprob_cls.prob.W[k] = prob_cls->prob.W[perm[j]];
			//printf("y %g x->index %d x->value %g w %g\n", subprob_cls.prob.y[k],subprob_cls.prob.x[k]->index,subprob_cls.prob.x[k]->value,subprob_cls.prob.W[k]);
			k++;
		}
		for (j=end_index; j<prob_cls->l; ++j) {
			subprob_cls.prob.y[k] = prob_cls->prob.y[perm[j]];
			subprob_cls.prob.x[k] = prob_cls->prob.x[perm[j]];
			subprob_cls.prob.W[k] = prob_cls->prob.W[perm[j]];
			//printf("y %g x->index %d x->value %g w %g\n", subprob_cls.prob.y[k],subprob_cls.prob.x[k]->index,subprob_cls.prob.x[k]->value,subprob_cls.prob.W[k]);
			k++;
		}

		// train
		int bag_size;
		struct stump** stump_bag;
		train_adaboost_stump(&subprob_cls, param, &stump_bag, &bag_size);
 
		// test
		double test_error = 0;
		for (j=start_index; j<end_index; ++j) {
			target_labels[perm[j]] = bag_predict_instance_label(prob_cls->prob.x[perm[j]], stump_bag, bag_size);
			if (target_labels[perm[j]] != prob_cls->prob.y[perm[j]])
				test_error += 1.0;
		}
		//test_error /= fold_size;
		//printf("error %g\n", test_error);
		printf("[fold-%d] (%d-%d) error %g (%d/%d)\n", i,start_index,end_index, test_error/fold_size, (int)test_error,fold_size);

		/*
		// construct test sub-problem
		struct problem test_prob;
		test_prob.l = fold_size;
		test_prob.n = prob_cls->n;
		test_prob.y = Malloc(double, test_prob.l);
		test_prob.x = Malloc(struct feature_node*, test_prob.l);
		test_prob.bias = prob_cls->prob.bias;
		test_prob.W = Malloc(double, test_prob.l);

		k = 0;
		for (j=start_index; j<end_index; ++j) {
			test_prob.y[k] = prob_cls->prob.y[perm[j]];
			test_prob.x[k] = prob_cls->prob.x[perm[j]];
			test_prob.W[k] = prob_cls->prob.W[perm[j]];
			k++;
		}

		double* aggr_labels = Malloc(double, fold_size);
		double cv_error = bag_predict_labels(&test_prob, stump_bag, bag_size, aggr_labels);
		for (j=0; j<fold_size; ++j)
			target_labels[perm[start_index+j]] = aggr_labels[j];
		free(aggr_labels);
		printf("%d-th fold cv error %g\n", i,cv_error);

		free(test_prob.y);
		free(test_prob.x);
		free(test_prob.W);
		*/

		for (j=0; j<bag_size; ++j)
			free(stump_bag[j]);
		free(stump_bag);
		free(subprob_cls.prob.y);
		free(subprob_cls.prob.x);
		free(subprob_cls.prob.W);
	}
	free(indices);
	free(perm);
}


/*
struct tree_node {
	struct stump* stump_;
	struct tree_node* left;
	struct tree_node* right;
};

struct tree_node* build_decision_tree(const struct problem_class* prob_class_col, int depth) {
	if (depth >= 2)
		return NULL;

	struct stump* stump_ = Malloc(struct stump,1);
	double error = find_best_stump(prob_class_col->prob, stump_);
	if (error <= 0.0) {
		free(stump_);
		return NULL;
	}

	struct tree_node* node = Malloc(struct tree_node,1);
	node->stump_ = stump_;

	double* pred_labels = Malloc(double,prob_class_col->prob->l);
	predict_label(prob_class_col->prob, pred_labels);
	//build sub-problem-class 1
	struct problem_class* subprob_class1 = Malloc(struct problem_class,1);
	// ...
	node->left = build_decision_tree(subprob_class1, depth+1);
	//build sub-problem-class 2
	struct problem_class* subprob_class2 = Malloc(struct problem_class,1);
	// ...
	node->right = build_decision_tree(sub_prob_class2, depth+1);

	// destrop sub-problem-class 1
	// destrop sub-problem-class 2
	free(pred_labels);
	return node;
}
*/

////////////////////////////////////////
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

#endif /* _STUMP_H */
