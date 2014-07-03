#include <stdlib.h>
#include <fftw3.h>

#define NUM_TESTS 

void generate_reference_data(int n, fftw_complex* in, fftw_complex* out){
  int i;
  fftw_plan p;

  for (i = 0; i < n; i++){
    in[i][0] = 2 * rand() / ((double)RAND_MAX + 1) - 1;
    in[i][1] = 2 * rand() / ((double)RAND_MAX + 1) - 1;
  }

  p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);  
}

void save_reference_data(int n, fftw_complex* in, fftw_complex* out){
  char s[100];
  FILE* f;

  sprintf(s, "fftw%d.in", n);
  f = fopen(s, "wb");
  fwrite(in, sizeof(fftw_complex), n, f);
  fclose(f);

  sprintf(s, "fftw%d.out", n);
  f = fopen(s, "wb");
  fwrite(out, sizeof(fftw_complex), n, f);
  fclose(f);
}

int main(int argc, char **argv){
  if (argc != 2){
    printf("Missing expected list of sizes");
    exit(0);
  }
  fftw_complex *in, *out;
  fftw_plan p;

  int n;

  FILE* f;
  f = fopen(argv[1], "r");
  while(!feof(f)){
    fscanf(f, "%i\n", &n);
    printf("Generating reference data for n = %i\n", n);
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    generate_reference_data(n, in, out);
    save_reference_data(n, in, out);
    fftw_free(in);
    fftw_free(out);
  }


  fclose(f);
}
