#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sys/stat.h>
#include <limits.h>
#include <float.h>
#include "util.h"
#include "linear.h"
#include "problem.h"
#include "adaboost-linear.h"
#define INF HUGE_VAL

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: train-adaboost-linear [options] training_set_file [model_file]\n"
	"options: (currently, it only supports binary classification)\n"
	"-a type : set type of AdaBoost solver (default 0)\n"
	"	 0 -- AdaBoost (binary)\n"
	"	 1 -- AdaBoost SAMME (multi-class)\n"
	"	 2 -- AdaBoost with one-against-all (OAA) decomposition (multi-class)\n"
	"-i iter : maximum number of AdaBoost iterations (default sqrt(#instances))\n"
	//"-r rate : sample rate of training data per AdaBoost iteration in (0.0,1.0] (default no sampling)\n"
	"-s type : set type of base solver (default 1)\n"
	"  for multi-class classification\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"  for regression\n"
	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n"
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
	"		where f is the primal function and pos/neg are # of\n"
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
	"	-s 1, 3, 4, and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight : weights adjust the parameter C of different classes (see README for details)\n"
	"-v n : n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
//	"-W instace_weight_file: initialize instance weights, (default 1.0/[#training instances])\n"
	);
	exit(1);
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
//void read_problem(const char *filename);
//void do_cross_validation();

struct problem_class prob_cls;
struct adaboost_linear_parameter adaparam;
struct adaboost_linear_model adamodel;
//struct feature_node *x_space;
struct parameter param;
//struct model* model_;
char* weight_file;
int flag_cross_validation;
int nr_fold;
double bias;

int main(int argc, char **argv)
{
	char input_file_name[1024] = {0};
	char model_file_name[1024] = {0};
	const char* error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem_class(input_file_name, &prob_cls, bias);

	error_msg = check_parameter(&prob_cls.prob,&param);
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	error_msg = check_adaboost_linear_input(&prob_cls,&adaparam);
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	print_adaboost_linear_parameter(&adaparam);
	print_problem_stats(&prob_cls);

	if(flag_cross_validation==1)
	{
		double* pred_labels = Malloc(double,prob_cls.l);
		cross_validate_adaboost_linear(&prob_cls, &adaparam, nr_fold, pred_labels);
		int i, correct=0;
		for (i=0; i<prob_cls.l; ++i)
			if (pred_labels[i] == prob_cls.prob.y[i])
				correct ++;
		double cv_error = 1.0-(double)correct/prob_cls.l;
		printf("Cross Validation Error %g\n", cv_error);
		free(pred_labels);
	}
	else
	{
		// train models
		train_adaboost_linear(&prob_cls, &adaparam, &adamodel);

		// save models
		if (save_adaboost_linear_model(model_file_name,&adamodel))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}

		// training error
		double* pred_labels = Malloc(double,prob_cls.l);
		double train_error = predict_adaboost_linear(&prob_cls.prob, &adamodel, pred_labels);
		printf("Training Error %g\n", train_error);
		free(pred_labels);

		// free model
		destroy_adaboost_linear_model(&adamodel);
	}

	destroy_param(&param);
	destroy_problem_class(&prob_cls);
	//free(x_space);

	return 0;
}

/*
void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);

	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR ||
	   param.solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
				((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
				((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			  );
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}

	free(target);
}
*/

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	adaparam.solver_type = ADABOOST;
	adaparam.max_iter = INT_MAX;
	adaparam.sample_rate = DBL_MAX;
	adaparam.linear_param = &param;

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	flag_cross_validation = 0;
	weight_file = NULL;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;

			case 'W':
				weight_file = argv[i];
				break;

			case 'a':
				adaparam.solver_type = atoi(argv[i]);
				break;

			case 'i':
				adaparam.max_iter = atoi(argv[i]);
				break;

			case 'r':
				adaparam.sample_rate = atof(argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.adamodel",p);
	}

	// make sure model file does not exist in training mode
	if (flag_cross_validation == 0) {
		struct stat s = {0};
		if (stat(model_file_name,&s) == 0) {
			fprintf(stderr, "Model file exists: %s\n", model_file_name);
			fprintf(stderr, "Please rename [model_file_name]\n");
			exit(1);
		}
	}

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
}

/*
// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
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
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	prob.W = Malloc(double,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
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

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);

	if(weight_file) 
	{
		fp = fopen(weight_file,"r");
		for(i=0;i<prob.l;i++)
			fscanf(fp,"%lf",&prob.W[i]);
		fclose(fp);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			prob.W[i] = 1.0/prob.l;
	}
}
*/
