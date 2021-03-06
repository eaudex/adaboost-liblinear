#include <stdlib.h>
#include <unistd.h>
#include "util.h"
#include "problem.h"
#include "linear.h"
#include "adaboost-linear.h"

void exit_with_help() {
	printf("Usage: predict-adaboost-linear [options] test_file model_file output_file\n"
	);
	exit(1);
}

void parse_command_line(int argc, char** argv, char* test_file_name, char* model_file_name, char* output_file_name);

int main(int argc, char** argv) {
	char test_file_name[1024];
	char model_file_name[1024];
	char output_file_name[1024];
	parse_command_line(argc, argv, test_file_name, model_file_name, output_file_name);

	// load models
	struct adaboost_linear_model adamodel;
	if (load_adaboost_linear_model(model_file_name,&adamodel) == -1) {
		fprintf(stderr, "can't load model file %s\n", model_file_name);
		exit(1);
	}

	// read problem
	double bias = -1.0;
	if (adamodel.model_bag[0]->bias >= 0.0)
		bias = adamodel.model_bag[0]->bias;
	struct problem_class prob_cls;
	read_problem_class(test_file_name, &prob_cls, bias);
	print_problem_stats(&prob_cls);

	double* pred_labels = Malloc(double, prob_cls.l);
	double test_error = predict_adaboost_linear(&prob_cls.prob, &adamodel, pred_labels);
	printf("test_error %lf\n", test_error);

	FILE* fp = fopen(output_file_name,"w");
	if(fp==NULL) return -1;
	int i;
	for (i=0; i<prob_cls.l; ++i)
		fprintf(fp, "%g\n", pred_labels[i]);
	if (ferror(fp)!=0 || fclose(fp)!=0) {
		fprintf(stderr, "can't save predicted labels %s\n", output_file_name);
		exit(1);
	}
	free(pred_labels);

	// free problem
	destroy_problem_class(&prob_cls);

	// free model
	destroy_adaboost_linear_model(&adamodel);

	return 0;
}


void parse_command_line(int argc, char** argv, char* test_file_name, char* model_file_name, char* output_file_name) {
	int c;
	while ((c=getopt(argc,argv,"")) != -1) {
		exit_with_help();
	}

	if (optind < argc)
		strcpy(test_file_name, argv[optind]);
	else
		exit_with_help();

	if (optind+1 < argc)
		strcpy(model_file_name, argv[optind+1]);
	else
		exit_with_help();

	if (optind+2 < argc)
		strcpy(output_file_name, argv[optind+2]);
	else
		exit_with_help();
}
