#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include "util.h"
#include "problem.h"
#include "linear.h"
#include "stump.h"

int to_test = 0;
int to_cross_validate = 0;
int num_folds = 2;
int quiet = 0;

void exit_with_help() {
	printf("Usage: train-adaboost-stump [options] train_file_name [model_file_name]\n"
		"options: (currently, it only supports binary classification)\n"
		"-i iter: maximum number of iterations (default sqrt(l)), where l denotes #training instances\n"
		"-d dimension: dimensions of random subspace; (default n, deterministic mode), where n denotes #features\n"
		"-t test_file_name: testing data after training\n"
		"-v n: n-fold cross validation mode\n"
		"-q: quiet mode\n"
	);
	exit(1);
}

void parse_command_line(int argc, char** argv, struct adaboost_stump_parameter* param, char* train_file_name, char* model_file_name, char* test_file_name);
double cross_validate();

int main(int argc, char** argv) {
	struct adaboost_stump_parameter param = {INT_MAX,INT_MAX};
	char train_file_name[1024] = {0};
	char model_file_name[1024] = {0};
	char test_file_name[1024] = {0};
	parse_command_line(argc, argv, &param, train_file_name, model_file_name, test_file_name);
	//printf("[train_file] %s\n", train_file_name);
	//printf("[model_file] %s\n", model_file_name);
	//if(to_test==1) printf("[test_file] %s\n", test_file_name);

	// read problem in libsvm format
	struct problem_class prob_cls;
	read_problem_class(train_file_name, &prob_cls, -1.0);
	print_problem_stats(&prob_cls);
	const char* error_msg = check_adaboost_input(&prob_cls, &param);
	if (error_msg) {
		fprintf(stderr, "INPUT ERROR: %s\n", error_msg);
		exit(1);
	}

	if (to_cross_validate) {
		double* pred_labels = Malloc(double,prob_cls.l);
		cross_validate_adaboost_stump(&prob_cls, &param, num_folds, pred_labels);
		int i, correct=0;
		for (i=0; i<prob_cls.l; ++i)
			if (pred_labels[i] == prob_cls.prob.y[i])
				correct ++;
		double cv_error = 1.0-(double)correct/prob_cls.l;
		printf("Cross Validation Error %g\n", cv_error);
		free(pred_labels);
	}
	else {
		// train models
		struct stump** stump_bag = NULL;
		int bag_size = 0;
		train_adaboost_stump(&prob_cls, &param, &stump_bag, &bag_size);

		// save models
		if (save_bag_stumps(model_file_name,stump_bag,bag_size)) {
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}

		// training error
		double* pred_labels = Malloc(double,prob_cls.l);
		double train_error = bag_predict_labels(&prob_cls.prob, stump_bag, bag_size, pred_labels);
		printf("Training Error %g\n", train_error);
		free(pred_labels);

		// test error
		if (to_test == 1) {
			struct problem_class test_prob_cls;
			read_problem_class(test_file_name, &test_prob_cls, -1);
			print_problem_stats(&test_prob_cls);
			//struct problem* test_prob = &test_prob_cls.prob;

			double* pred_labels = Malloc(double,test_prob_cls.l);
			double test_error = bag_predict_labels(&test_prob_cls.prob,stump_bag,bag_size,pred_labels);
			printf("Test Error %g\n", test_error);

			free(pred_labels);
			destroy_problem_class(&test_prob_cls);
		}

		// free models
		int i;
		for (i=0; i<bag_size; ++i)
			free(stump_bag[i]);
		free(stump_bag);
	}

	destroy_problem_class(&prob_cls);
	return 0;
}


//TODO
double cross_validate() {
	// split problems into n folds
	// for each fold
	//	train_adaboost_stump on the other folds
	//	bag_predict_labels0 on the fold
	// aggregate predicted accuracy
	return 0.0;
}

void parse_command_line(int argc, char** argv, struct adaboost_stump_parameter* param, char* train_file_name, char* model_file_name, char* test_file_name) {
	int c;
	while ((c=getopt(argc,argv,"i:d:t:v:q")) != -1) {
		switch (c) {
			case 'i':
				param->max_iter = atoi(optarg);
				printf("[max_iter] %d\n", param->max_iter);
				break;
			case 'd':
				param->random_subspace = atoi(optarg);
				printf("[random_subspace] %d\n", param->random_subspace);
				break;
			case 't':
				to_test = 1;
				strcpy(test_file_name, optarg);
				//printf("[test_file] %s\n", test_file_name);
				break;
			case 'v':
				to_cross_validate = 1;
				num_folds = atoi(optarg);
				printf("[cross_validation] %d folds\n", num_folds);
				break;
			case 'q':
				quiet = 1;
				printf("[quiet]\n");
				break;
		}
	}

	if (optind < argc)
		strcpy(train_file_name, argv[optind]);
	else
		exit_with_help();
	if (optind+1 < argc)
		strcpy(model_file_name, argv[optind+1]);
	else {
		char* p = strrchr(argv[optind],'/');
		if (p == NULL)
			p = argv[optind];
		else
			p += 1;
		sprintf(model_file_name, "%s.adamodel",p);
	}
}

